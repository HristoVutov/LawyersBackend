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
    limit: int | None = None,
    conversation_id: str = "default"
) -> str:
    """
    Search through indexed documents.
    NOTE: This tool must be executed on the client-side via the Remote Bridge.
    """
    return f"Error: search_indexed must be executed on the client-side. Tool bridge not active for query: {query}"


# Export all tools
SEARCH_TOOLS = [search_indexed]
