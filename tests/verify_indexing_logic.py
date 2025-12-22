import asyncio
import sys
import os
from pathlib import Path

from app.services.indexer import index_directory
from app.agents import initialize_agent_registry

async def main():
    initialize_agent_registry()
    test_dir = r"C:\Users\vutov\Documents\PersonalProjects\Lawyers\LawyersBackend\tests\data_to_index"
    print(f"Starting indexing for: {test_dir}")
    await index_directory(test_dir)
    print("Indexing completed.")

if __name__ == "__main__":
    asyncio.run(main())
