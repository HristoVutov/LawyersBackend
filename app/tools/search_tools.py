"""
Search Tools - Semantic and keyword search for indexed documents.

Provides:
- search_indexed: Hybrid search (vector + keyword) through indexed documents
- Entity extraction and memory from search results
- Search result caching
"""
import json
import math
from pathlib import Path
from langchain_core.tools import tool

from app.config import get_settings
from app.middleware.virtual_fs import get_vfs, resolve_path
from app.services import (
    session_store,
    cache_search_results,
    get_cached_search,
    remember_entity,
    find_entity,
)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def extract_entities_from_results(query: str, results: list[dict]) -> list[dict]:
    """
    Extract potential entities (companies, people) from search query and results.
    Focused on Bulgarian legal documents.
    """
    entities = []
    
    # Common Bulgarian company suffixes
    company_suffixes = ["ООД", "ЕООД", "АД", "ЕАД", "ДЗЗД", "СД", "КД", "КДА"]
    
    query_upper = query.upper()
    
    # Check query for company patterns
    for suffix in company_suffixes:
        if suffix in query_upper:
            import re
            pattern = rf"([А-ЯA-Z\s]+\s*{suffix})"
            match = re.search(pattern, query_upper, re.IGNORECASE)
            if match:
                entities.append({
                    "name": match.group(1).strip(),
                    "type": "company",
                    "source": "query"
                })
    
    # Check each result for entities
    for result in results:
        keywords = (result.get("keywords") or "").upper()
        summary = (result.get("summary") or "").upper()
        
        # Look for company names in keywords
        for suffix in company_suffixes:
            import re
            pattern = rf'"?([А-ЯA-Z\s]+\s*{suffix})"?'
            matches = re.findall(pattern, keywords, re.IGNORECASE)
            for m in matches:
                entities.append({
                    "name": m.replace('"', "").strip(),
                    "type": "company",
                    "source": "document",
                    "related_doc": result.get("original_file")
                })
        
        # Look for BULSTAT numbers
        bulstat_match = re.search(r"БУЛСТАТ[:\s]*(\d+)", summary, re.IGNORECASE)
        if not bulstat_match:
            bulstat_match = re.search(r"БУЛСТАТ[:\s]*(\d+)", keywords, re.IGNORECASE)
        
        if bulstat_match:
            entities.append({
                "name": f"Company BULSTAT {bulstat_match.group(1)}",
                "type": "company",
                "identifiers": {"bulstat": bulstat_match.group(1)},
                "source": "document",
                "related_doc": result.get("original_file")
            })
    
    # Deduplicate by name
    seen = set()
    unique_entities = []
    for entity in entities:
        key = entity["name"].upper()
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    return unique_entities


async def get_query_embedding(query: str) -> list[float] | None:
    """Get embedding for a query using Gemini API."""
    try:
        import google.generativeai as genai
        
        settings = get_settings()
        if not settings.google_api_key:
            return None
        
        genai.configure(api_key=settings.google_api_key)
        
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query
        )
        return result["embedding"]
    except Exception as e:
        print(f"[SEARCH] Embedding failed: {e}")
        return None


@tool
async def search_indexed(
    query: str,
    search_type: str = "hybrid",
    limit: int = 10,
    conversation_id: str = "default"
) -> str:
    """
    Search through indexed documents using semantic/vector search and keywords.
    Use this to find relevant legal documents.
    
    Args:
        query: Search query - keywords or description of what you're looking for
        search_type: Search type: 'keyword', 'vector', or 'hybrid' (default)
        limit: Max results to return (default 10)
        conversation_id: Conversation ID for caching
    """
    try:
        settings = get_settings()
        
        # ========== CHECK CACHE FIRST ==========
        cached = await get_cached_search(session_store, conversation_id, query)
        if cached:
            print(f'[SEARCH] Cache hit for: "{query}"')
            return f"[CACHED RESULTS]\n{json.dumps(cached, ensure_ascii=False)}"
        
        # Check if query matches a known entity
        entity = await find_entity(session_store, conversation_id, query)
        if entity and entity.get("related_docs"):
            print(f'[SEARCH] Entity match found: {entity["name"]}')
            return f"""[ENTITY MEMORY]
Found known entity: {entity['name']}
Type: {entity.get('type', 'unknown')}
Related documents: {', '.join(entity.get('related_docs', []))}

Use add_to_context with these files for more details."""
        
        # Find all .index.json files using virtual file system
        # settings.indexed_files_dir now correctly points to .indexedfiles in project or global
        vfs = get_vfs()
        
        # STRICT MODE: Must have project path
        if not vfs._project_path:
             return "ERROR: No active case folder selected. Please open a case folder before searching."
             
        search_dirs = [vfs._project_path / ".indexedfiles"]
        
        if vfs._project_path:
             search_dirs.append(vfs._project_path)
        
        index_files = []
        for search_dir in search_dirs:
            try:
                if search_dir.exists():
                    for f in search_dir.iterdir():
                        if f.name.endswith(".index.json"):
                            index_files.append(f)
            except Exception:
                pass
        
        if not index_files:
            return "No indexed files (.index.json) found in project folder."
        
        # Get query embedding for vector search
        query_embedding = None
        if search_type in ("hybrid", "vector"):
            query_embedding = await get_query_embedding(query)
        
        results = []
        query_lower = query.lower()
        terms = [t for t in query_lower.split() if len(t) > 2]
        
        for index_file in index_files:
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                keyword_score = 0
                semantic_score = 0.0
                matched_fields = []
                
                # Keyword search
                if search_type in ("keyword", "hybrid"):
                    content_lower = (data.get("content") or "").lower()
                    summary_lower = (data.get("summary") or "").lower()
                    
                    for term in terms:
                        if term in content_lower:
                            keyword_score += 3
                        if term in summary_lower:
                            keyword_score += 5
                        if data.get("keywords"):
                            for kw in data["keywords"]:
                                if term in kw.lower():
                                    keyword_score += 8
                                    matched_fields.append("keywords")
                        if data.get("documentType") and term in data["documentType"].lower():
                            keyword_score += 5
                            matched_fields.append("docType")
                    
                    if keyword_score > 0:
                        matched_fields.append("text")
                
                # Vector/Semantic search
                if query_embedding and data.get("embedding") and search_type in ("hybrid", "vector"):
                    similarity = cosine_similarity(query_embedding, data["embedding"])
                    if similarity > 0.22:
                        semantic_score = similarity * 50
                        matched_fields.append(f"vector({similarity:.2f})")
                
                total_score = keyword_score + semantic_score
                
                if total_score > 0:
                    results.append({
                        "original_file": index_file.name.replace(".index.json", ""),
                        "summary": data.get("summary", "No summary"),
                        "document_type": data.get("documentType", "Unknown"),
                        "keywords": ", ".join((data.get("keywords") or [])[:5]),
                        "score": total_score,
                        "matched_fields": list(set(matched_fields))
                    })
            except Exception:
                pass
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Dynamic thresholding
        max_score = results[0]["score"] if results else 0
        
        if max_score > 80:
            threshold = 65
        elif max_score > 60:
            threshold = 40
        else:
            threshold = 1
        
        top_results = [r for r in results if r["score"] >= threshold][:limit]
        
        if not top_results:
            return f"No documents found matching: {query}"
        
        # Format results
        formatted = []
        for r in top_results:
            formatted.append(
                f"📄 {str(vfs._project_path / r['original_file']) if vfs._project_path else r['original_file']}\n"
                f"   Type: {r['document_type']}\n"
                f"   Summary: {r['summary'][:150]}...\n"
                f"   Keywords: {r['keywords']}\n"
                f"   Score: {r['score']:.1f} [{', '.join(r['matched_fields'])}]"
            )
        
        formatted_results = "\n\n".join(formatted)
        
        # Cache results
        await cache_search_results(session_store, conversation_id, query, top_results)
        print(f'[SEARCH] Cached results for: "{query}"')
        
        # Extract and remember entities
        extracted_entities = extract_entities_from_results(query, top_results)
        for entity in extracted_entities:
            await remember_entity(
                session_store,
                conversation_id,
                entity["name"],
                {
                    "type": entity["type"],
                    "identifiers": entity.get("identifiers", {}),
                    "related_docs": [entity["related_doc"]] if entity.get("related_doc") else [r["original_file"] for r in top_results],
                    "aliases": []
                }
            )
            print(f'[ENTITY] Remembered: {entity["name"]} ({entity["type"]})')
        
        return formatted_results
        
    except Exception as e:
        return f"Error searching indexed files: {str(e)}"


# Export all tools
SEARCH_TOOLS = [search_indexed]
