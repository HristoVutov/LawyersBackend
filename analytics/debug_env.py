from pathlib import Path
import os
from dotenv import load_dotenv

p = Path("app/config.py").resolve()
print(f"Script path: {p}")
env_path = p.parent.parent / ".env"
print(f"Calculated env path: {env_path}")
print(f"Env file exists? {env_path.exists()}")

print("--- Pre-load env ---")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")

load_dotenv(env_path)

print("--- Post-load env ---")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
