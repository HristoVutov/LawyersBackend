
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os

# Hardcoded for simplicity in verification script, or could read .env
DATABASE_URL = "postgresql+asyncpg://postgres:1234@localhost:5432/lawyers_analytics"

async def verify_payment():
    engine = create_async_engine(DATABASE_URL)
    
    async with engine.connect() as conn:
        print("\n--- Transactions (Most Recent First) ---")
        result = await conn.execute(text("SELECT id, status, amount_cents, credits_amount FROM transactions ORDER BY created_at DESC LIMIT 5"))
        rows = result.fetchall()
        if not rows:
            print("No transactions found.")
        else:
            for row in rows:
                print(f"ID: {row.id} | Status: {row.status} | Amount: {row.amount_cents/100:.2f} | Credits: {row.credits_amount}")

        print("\n--- User Credits ---")
        result = await conn.execute(text("SELECT email, available_credits FROM users"))
        rows = result.fetchall()
        for row in rows:
            print(f"User: {row.email} | Available Credits: {row.available_credits}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(verify_payment())
