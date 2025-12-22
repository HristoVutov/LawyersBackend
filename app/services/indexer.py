import json
import asyncio
import aiofiles
from pathlib import Path
from app.config import get_settings
from app.services.file_readers import read_any_file
from app.agents.analysis_agent import AnalysisAgent
# You might reuse the embedding logic from search_tools or centralized it
import google.generativeai as genai

async def generate_embedding(text: str):
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result["embedding"]

async def index_file(file_path: Path, project_root: Path):
    settings = get_settings()
    
    # 0. STRICT MODE: Ensure active project
    # Import here to avoid circular deprecation
    from app.middleware.virtual_fs import get_vfs
    vfs = get_vfs()
    
    if not vfs._project_path:
        raise ValueError("STRICT MODE EXCEPTION: No active project set. Cannot index files.")
        
    """
    1. Read file content
    2. Analyze with AnalysisAgent
    3. Generate embedding
    4. Save .index.json
    """
    print(f"[Indexer] Processing {file_path.name}...")
    
    # Check for existing index to skip if unmodified
    index_dir = settings.indexed_files_dir
    index_file_path = index_dir / f"{file_path.name}.index.json"
    
    if index_file_path.exists():
        source_mtime = file_path.stat().st_mtime
        index_mtime = index_file_path.stat().st_mtime
        
        # If index is newer than source, check if summary is valid
        if index_mtime > source_mtime:
            # Check if summary exists and is not null/empty/failed
            try:
                with open(index_file_path, 'r', encoding='utf-8') as f:
                    existing_index = json.load(f)
                summary = existing_index.get('summary')
                # Check for valid summary (not null, not empty, not error fallback)
                if summary and len(str(summary).strip()) > 0 and summary != "Analysis failed to parse":
                    print(f"[Indexer] Skipped {file_path.name} (already indexed and valid)")
                    return
                else:
                    print(f"[Indexer] Re-analyzing {file_path.name} (summary is null/empty/failed)")
            except Exception as e:
                print(f"[Indexer] Re-indexing {file_path.name} (failed to read existing index: {e})")

    # 1. Read
    content = await read_any_file(file_path)
    if not content or len(content) < 10:
        print(f"[Indexer] Skipped {file_path.name} (empty/unreadable)")
        return

    # 2. Analyze (Using the AnalysisAgent we created)
    analysis_agent = AnalysisAgent()
    # BaseAgent.invoke returns the final string response
    analysis_json_str = await analysis_agent.invoke(
        f"Analyze this document:\n\n{content[:15000]}" # Truncate for safety
    )
    
    try:
        analysis_data = json.loads(analysis_json_str)
    except json.JSONDecodeError:
        # Agent returned text instead of JSON - use the text as the summary
        print(f"[Indexer] Agent returned text (not JSON), using as summary")
        # Extract basic keywords from the first few sentences
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', analysis_json_str[:1000])
        keywords = list(set(words))[:10]  # Unique, max 10
        analysis_data = {
            "summary": analysis_json_str[:5000],  # Use response as summary (truncate if needed)
            "keywords": keywords,
            "documentType": "Unknown"
        }

    # 3. Embed (Use content truncated to 20k chars as per documentation)
    # The documentation requires the extracted text content, not just summary
    text_to_embed = content[:20000] if content else ""
    embedding = await generate_embedding(text_to_embed)

    # 4. Save (truncate content in log output)
    print(f"[Indexer] Saving {file_path.name} (content: {content[:25]}...)")
    index_data = {
        **analysis_data,
        "content": content, # Optional: save full content?
        "embedding": embedding,
        "original_file": file_path.name,
        "last_modified": file_path.stat().st_mtime
    }
    
    # Save to indexedfiles directory (CaseFolder/.indexedfiles)
    index_dir = settings.indexed_files_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    
    index_file_path = index_dir / f"{file_path.name}.index.json"
    
    async with aiofiles.open(index_file_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(index_data, ensure_ascii=False, indent=2))
        
    print(f"[Indexer] Saved index for {file_path.name} to {index_file_path}")

async def index_directory(directory_path: str):
    p = Path(directory_path)
    # Import here to avoid circular checks if not needed at module level
    from app.middleware.virtual_fs import get_vfs
    vfs = get_vfs()
    
    if not vfs._project_path:
        print("[Indexer] STRICT MODE WARNING: No active project set. Indexing might fail or go to wrong place.")

    settings = get_settings()
    
    # 0. Discovery Phase
    print(f"[Indexer] Scanning directory {directory_path}...")
    all_files = []
    # Use the tuple directly to ensure compatibility if SUPPORTED_EXTENSIONS is defined later
    exts = ('.txt', '.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg')
    
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            if not f.name.endswith(".index.json"):
                all_files.append(f)

    # 1. Filter & Read/OCR Phase (One by one)
    # Store data in memory: [ { "path": Path, "content": str, "embedding": list, "analysis": dict } ]
    work_queue = [] 

    print(f"[Indexer] Starting Phase 1: Check & Read/OCR ({len(all_files)} candidates)...")
    for f in all_files:
        # -- Incremental Check Logic --
        index_dir = settings.indexed_files_dir
        index_file_path = index_dir / f"{f.name}.index.json"
        
        needs_indexing = True
        if index_file_path.exists():
            try:
                source_mtime = f.stat().st_mtime
                index_mtime = index_file_path.stat().st_mtime
                
                if index_mtime > source_mtime:
                     with open(index_file_path, 'r', encoding='utf-8') as json_f:
                        existing_index = json.load(json_f)
                        summary = existing_index.get('summary')
                        if summary and len(str(summary).strip()) > 0 and summary != "Analysis failed to parse":
                            needs_indexing = False
            except Exception as e:
                print(f"[Indexer] Error checking index for {f.name}: {e}")
                needs_indexing = True 

        if not needs_indexing:
             print(f"[Indexer] Skipped {f.name} (already indexed)")
             continue

        # -- Read / OCR --
        print(f"[Indexer] Reading/OCR {f.name}...")
        try:
            content = await read_any_file(f)
            if content and len(content) >= 10:
                work_queue.append({
                    "path": f,
                    "content": content,
                    "start_mtime": f.stat().st_mtime
                })
            else:
                print(f"[Indexer] Skipped {f.name} (empty/unreadable)")
        except Exception as e:
             print(f"[Indexer] Failed to read {f.name}: {e}")

    # 2. Embedding Phase (One by one)
    print(f"[Indexer] Starting Phase 2: Embedding ({len(work_queue)} files)...")
    for item in work_queue:
        print(f"[Indexer] Generating embedding for {item['path'].name}...")
        try:
            text_to_embed = item["content"][:20000] # Truncate for API limits
            embedding = await generate_embedding(text_to_embed)
            item["embedding"] = embedding
        except Exception as e:
            print(f"[Indexer] Embedding failed for {item['path'].name}: {e}")
            item["embedding"] = [] 

    # 3. Analysis & Save Phase (One by one)
    print(f"[Indexer] Starting Phase 3: Analysis & Save ({len(work_queue)} files)...")
    analysis_agent = AnalysisAgent()
    
    for item in work_queue:
        f_path = item["path"]
        content = item["content"]
        embedding = item["embedding"]
        
        print(f"[Indexer] Analyzing {f_path.name}...")
        try:
            # -- Analysis --
            analysis_json_str = await analysis_agent.invoke(
                f"Analyze this document:\n\n{content[:15000]}"
            )
            
            try:
                analysis_data = json.loads(analysis_json_str)
            except json.JSONDecodeError:
                print(f"[Indexer] Agent returned text (not JSON), using as summary for {f_path.name}")
                import re
                words = re.findall(r'\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b', analysis_json_str[:1000])
                keywords = list(set(words))[:10]
                analysis_data = {
                    "summary": analysis_json_str[:5000],
                    "keywords": keywords,
                    "documentType": "Unknown"
                }

            # -- Save --
            print(f"[Indexer] Saving {f_path.name}...")
            index_data = {
                **analysis_data,
                "content": content,
                "embedding": embedding,
                "original_file": f_path.name,
                "last_modified": item["start_mtime"]
            }
            
            index_dir = settings.indexed_files_dir
            index_dir.mkdir(parents=True, exist_ok=True)
            out_path = index_dir / f"{f_path.name}.index.json"
            
            async with aiofiles.open(out_path, 'w', encoding='utf-8') as out_f:
                await out_f.write(json.dumps(index_data, ensure_ascii=False, indent=2))
                
            print(f"[Indexer] Saved {out_path.name}")

        except Exception as e:
            print(f"[Indexer] Analysis/Save failed for {f_path.name}: {e}")

    print("[Indexer] Batch processing complete.")


SUPPORTED_EXTENSIONS = ('.txt', '.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg')


def get_indexing_status(project_path: Path) -> dict:
    """
    Calculate the indexing status for a project directory.
    
    Returns:
        dict with total_files, indexed_files, stale_files, and status
    """
    settings = get_settings()
    index_dir = settings.indexed_files_dir
    
    total_files = 0
    indexed_files = 0
    stale_files = 0
    
    for f in project_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # Skip files in hidden directories like .indexedfiles, .generatedFiles
        if any(part.startswith('.') for part in f.relative_to(project_path).parts):
            continue
            
        total_files += 1
        
        index_file_path = index_dir / f"{f.name}.index.json"
        if index_file_path.exists():
            source_mtime = f.stat().st_mtime
            index_mtime = index_file_path.stat().st_mtime
            
            if index_mtime > source_mtime:
                indexed_files += 1
            else:
                stale_files += 1
    
    # Determine status
    if total_files == 0:
        status = "empty"
    elif indexed_files == total_files:
        status = "fully_indexed"
    elif indexed_files == 0:
        status = "not_indexed"
    else:
        status = "partially_indexed"
    
    return {
        "total_files": total_files,
        "indexed_files": indexed_files,
        "stale_files": stale_files,
        "status": status,
    }
