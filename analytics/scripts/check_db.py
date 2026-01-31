import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Add the parent directory to sys.path to import app modules
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings

async def check_connection():
    settings = get_settings()
    print(f"Checking connection to: {settings.database_url}")
    
    try:
        engine = create_async_engine(settings.database_url)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            print(f"Result: {result.scalar()}")
        await engine.dispose()
    except Exception as e:
        print("❌ Database connection failed.")
        print(f"Error: {e}")
        # Suggest probable causes
        if "ConnectionRefusedError" in str(e):
            print("\nPossible cause: PostgreSQL is not running or not listening on the specified port.")
        elif "InvalidCatalogNameError" in str(e) or 'database "lawyers_analytics" does not exist' in str(e):
            print("\nPossible cause: The database 'lawyers_analytics' does not exist.")
            print("To fix, run: createdb lawyers_analytics")
        elif "password authentication failed" in str(e):
            print("\nPossible cause: Incorrect username or password in DATABASE_URL.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_connection())
