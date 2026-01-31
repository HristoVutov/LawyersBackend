import asyncio
import sys
import asyncpg

async def create_database():
    try:
        # Connect to default 'postgres' database to create new db
        conn = await asyncpg.connect('postgresql://postgres:1234@localhost:5432/postgres')
        
        # Check if exists
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = 'lawyers_analytics'")
        if not exists:
            await conn.execute('CREATE DATABASE lawyers_analytics')
            print("✅ Database 'lawyers_analytics' created successfully.")
        else:
            print("ℹ️ Database 'lawyers_analytics' already exists.")
            
        await conn.close()
    except Exception as e:
        print(f"❌ Error creating database: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(create_database())
