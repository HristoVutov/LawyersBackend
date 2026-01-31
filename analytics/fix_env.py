
content = """DATABASE_URL=postgresql+asyncpg://postgres:1234@localhost:5432/lawyers_analytics
API_KEY=dev-secret-key
HOST=0.0.0.0
PORT=8001
CORS_ORIGINS=*
"""
with open(".env", "w", encoding="utf-8") as f:
    f.write(content)
print("Using python to write Clean .env file written.")
