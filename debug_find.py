import json
from pathlib import Path

base_dir = Path(r"C:\Users\vutov\Documents\NLP\SplitTest\.indexedfiles")
parent_path = base_dir / "c847fd1a-0129-49d3-923c-6c3e1eabf2ad.pdf.index.json"

with open(parent_path, 'r', encoding='utf-8') as f:
    parent_data = json.load(f)

full_content = parent_data.get("content", "")
segments = parent_data.get("documentSegments", [])

seg2 = segments[1] # Segment 2
start_text = seg2.get("startText", "")

print(f"Searching for: '{start_text}'")
print(f"In content (first 200 chars): {full_content[:200]}")

idx = full_content.find(start_text)
print(f"Direct Find Result: {idx}")

# Try normalization
print("\nNormalized check:")
norm_content = " ".join(full_content.split())
norm_start = " ".join(start_text.split())
idx_norm = norm_content.find(norm_start)
print(f"Normalized Find Result: {idx_norm}")
