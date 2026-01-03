import requests
import json

base_url = "http://127.0.0.1:8000/api"

def set_project():
    url = f"{base_url}/project"
    payload = {"project_path": "C:\\Users\\vutov\\Documents\\NLP\\SplitTest"}
    try:
        response = requests.post(url, json=payload)
        print(f"[Setup] Set Project: {response.status_code}")
    except Exception as e:
        print(f"[Setup] Failed: {e}")

def test_split():
    # Only test file_path trigger, relying on backend index lookup
    url = f"{base_url}/files/split"
    payload = {
        "file_path": "c847fd1a-0129-49d3-923c-6c3e1eabf2ad.pdf.index.json"
    }
    resp = requests.post(url, json=payload)
    print(f"Split Status: {resp.status_code}")
    print(f"Split Body: {resp.text}")

set_project()
test_split()
