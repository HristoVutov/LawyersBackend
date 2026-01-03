import json
import asyncio
import aiofiles
from pathlib import Path
from app.config import get_settings
from app.services.file_readers import read_any_file
from app.agents.analysis_agent import AnalysisAgent
from app.agents.schemas.analysis_schemas import DocumentSegment
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
    
    try:
        # Use structured output method
        analysis_result = await analysis_agent.analyze(content[:200000])
        # Convert Pydantic model to dict
        analysis_data = analysis_result.model_dump()
    except Exception as e:
        print(f"[Indexer] Analysis failed: {e}")
        # Fallback
        analysis_data = {
            "summary": "Analysis failed", 
            "keywords": [], 
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
            # -- PRE-SAVE (Pending) --
            # Save a placeholder so the file exists even if analysis crashes/hangs
            print(f"[Indexer] Saving pending index for {f_path.name}...")
            pending_index_data = {
                "documentType": "Analyzing",
                "containsMultipleDocuments": False,
                "documentSegments": [],
                "summary": "Analysis in progress... (Process started)",
                "keywords": [],
                "parties": [],
                "keyClauses": [],
                "risks": [],
                "missingElements": [],
                "indexFields": {},
                "overallScore": 0,
                "content": content,
                "embedding": embedding,
                "original_file": f_path.name,
                "last_modified": item["start_mtime"]
            }
            
            index_dir = settings.indexed_files_dir
            index_dir.mkdir(parents=True, exist_ok=True)
            out_path = index_dir / f"{f_path.name}.index.json"
            
            async with aiofiles.open(out_path, 'w', encoding='utf-8') as out_f:
                await out_f.write(json.dumps(pending_index_data, ensure_ascii=False, indent=2))

            # -- Analysis --
            try:
                analysis_result = await analysis_agent.analyze(content[:200000])
                analysis_data = analysis_result.model_dump()
            except Exception as e:
                print(f"[Indexer] Analysis failed for {f_path.name}: {e}")
                analysis_data = {
                    "summary": "Analysis failed",
                    "keywords": [],
                    "documentType": "Unknown"
                }

            # -- Save Final --
            print(f"[Indexer] Saving final index for {f_path.name}...")
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


def find_best_match_location(content: str, query: str) -> int:
    """
    Finds the best starting index of 'query' inside 'content' using heuristics.
    Returns -1 if completely lost.
    """
    if not query:
        return -1
        
    # 1. Exact Match
    idx = content.find(query)
    if idx != -1:
        return idx
        
    # 2. Normalized Match (ignore whitespace noise)
    # This is expensive to map back, so we use it just to check existence mostly,
    # but let's try a simpler approach: splitting into chunks.
    
    # 3. Chunk Match (Try finding the first 20 chars, or middle 20, etc)
    # Often the LLM hallucinates the end of the phrase but gets the start right, or vice versa.
    
    # Try first 30 chars
    if len(query) > 30:
        chunk = query[:30]
        idx = content.find(chunk)
        if idx != -1:
            return idx
            
    # Try last 30 chars
    if len(query) > 30:
        chunk = query[-30:]
        idx = content.find(chunk)
        if idx != -1:
            # We found the end of the query string.
            # We want the start, so we subtract length (approx)
            return max(0, idx - (len(query) - 30))

    # Try middle
    if len(query) > 60:
        mid = len(query) // 2
        chunk = query[mid:mid+30]
        idx = content.find(chunk)
        if idx != -1:
            return max(0, idx - mid)
            
    # 4. Words Match (First 5 words)
    words = query.split()
    if len(words) > 5:
        chunk = " ".join(words[:5])
        idx = content.find(chunk)
        if idx != -1:
            return idx
            
    return -1


async def create_virtual_index(parent_file_path: str, segment_index: int, segment_data: DocumentSegment):
    """
    Create a virtual index file for a segment of a document.
    
    Args:
        parent_file_path: The name of the original file (e.g. Contract.pdf)
        segment_index: The index of the segment (1-based)
        segment_data: The segment data (summary, type, start/end text)
    """
    settings = get_settings()
    index_dir = settings.indexed_files_dir
    
    # 1. Load Parent Index to get content context
    parent_index_path = index_dir / f"{parent_file_path}.index.json"
    if not parent_index_path.exists():
        raise FileNotFoundError(f"Parent index not found: {parent_index_path}")
        
    async with aiofiles.open(parent_index_path, 'r', encoding='utf-8') as f:
        parent_data = json.loads(await f.read())
        
    full_content = parent_data.get("content", "")
    
    # 2. Extract Segment Content
    # We use startText and endText to find the slice.
    # This is a heuristic: finding the first occurrence of startText 
    # and the last occurrence of endText after startText.
    
    # Strategy 0: Page-Based Slicing (Priority)
    start_idx = -1
    end_idx = -1
    
    if segment_data.segmentStartPage is not None:
        page_marker = f"[Page {segment_data.segmentStartPage}]"
        start_idx = full_content.find(page_marker)
        if start_idx == -1:
             print(f"[Indexer] Warning: Start page {segment_data.segmentStartPage} marker not found.")
    
    # Strategy 1: Text-Based Slicing (Fallback)
    if start_idx == -1:
        start_idx = find_best_match_location(full_content, segment_data.startText)
        if start_idx == -1:
            print(f"[Indexer] Warning: Start text not found for segment {segment_index}, defaulting to 0")
            start_idx = 0

    # Determine End Index
    if segment_data.segmentEndPage is not None:
         # Try to find the start of the NEXT page to include the full end page
         next_page_marker = f"[Page {segment_data.segmentEndPage + 1}]"
         end_idx = full_content.find(next_page_marker, start_idx)
         
         if end_idx == -1:
             # If next page not found (maybe last page), try finding the marker of the end page itself
             # But that would cut the content of the end page unless we go to next marker.
             # So we might fallback to finding the end of the file or text match.
             pass

    if end_idx == -1:
         # Search for end text AFTER start text
        if segment_data.endText:
            end_match = find_best_match_location(full_content[start_idx:], segment_data.endText)
            if end_match != -1:
                end_idx = start_idx + end_match + len(segment_data.endText)
            else:
                 end_idx = len(full_content)
        else:
            end_idx = len(full_content)
            
    segment_content = full_content[start_idx:end_idx]
    
    # 3. Generate Embedding for the segment
    # (Optional: we could re-use parent embedding or portions, but new is better)
    try:
        embedding = await generate_embedding(segment_content[:20000])
    except Exception:
        embedding = []

    analysis_agent = AnalysisAgent()
    try:
        print(f"[Indexer] Running full analysis on segment {segment_index}...")
        segment_analysis = await analysis_agent.analyze(segment_content[:200000])
        # Convert pydantic to dict
        segment_analysis_dict = segment_analysis.model_dump()
        
        # Override fields with analysis results
        keywords = segment_analysis_dict.get("keywords", [])
        parties = segment_analysis_dict.get("parties", [])
        risks = segment_analysis_dict.get("risks", [])
        summary = segment_analysis_dict.get("summary") or segment_data.summary
        doc_type = segment_analysis_dict.get("documentType") or segment_data.segmentType
        
    except Exception as e:
        print(f"[Indexer] Warning: Segment analysis failed: {e}")
        # Fallback to basic data
        keywords = []
        parties = []
        risks = []
        summary = segment_data.summary
        doc_type = segment_data.segmentType

    # 4. Create Virtual Index Data
    virtual_filename = f"{parent_file_path}.segment{segment_index}"
    
    virtual_index_data = {
        "documentType": doc_type,
        "summary": summary,
        "keywords": keywords, 
        "parties": parties, 
        "risks": risks,
        "content": segment_content,
        "embedding": embedding,
        "original_file": parent_file_path, 
        "is_virtual": True,
        "virtual_source": virtual_filename,
        "last_modified": parent_data.get("last_modified")
    }
    
    # 5. Save
    virtual_index_path = index_dir / f"{virtual_filename}.index.json"
    
    async with aiofiles.open(virtual_index_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(virtual_index_data, ensure_ascii=False, indent=2))
        
    return str(virtual_index_path)


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
