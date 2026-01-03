import json
from pathlib import Path

base_dir = Path(r"C:\Users\vutov\Documents\NLP\SplitTest\.indexedfiles")
seg2_path = base_dir / "c847fd1a-0129-49d3-923c-6c3e1eabf2ad.pdf.segment2.index.json"
seg3_path = base_dir / "c847fd1a-0129-49d3-923c-6c3e1eabf2ad.pdf.segment3.index.json"

def read_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("content", ""), data.get("summary", "")

c2, s2 = read_content(seg2_path)
c3, s3 = read_content(seg3_path)

print(f"Segment 2 Length: {len(c2)}")
print(f"Segment 2 Start: {c2[:30]}...")
print(f"Segment 2 Summary: {s2[:50]}...")
print("-" * 20)
print(f"Segment 3 Length: {len(c3)}")
print(f"Segment 3 Start: {c3[:30]}...")
print(f"Segment 3 Summary: {s3[:50]}...")

if c2 == c3:
    print("WARNING: CONTENTS ARE IDENTICAL!")
else:
    print("SUCCESS: Contents are different.")
