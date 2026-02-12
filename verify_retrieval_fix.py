import asyncio
from app.tools.research_tools import get_legal_references

async def verify_tool():
    query = "ЗЗД нищожност договори"
    print(f"--- Testing get_legal_references with query: '{query}' ---")
    
    try:
        result = await get_legal_references.ainvoke(query)
        print("Result received.\n")
        print(result[:500] + "..." if len(result) > 500 else result)
        
        if "article_content" in result:
            print("\n✅ SUCCESS: 'article_content' field found in JSON output.")
        else:
            print("\n❌ FAILURE: 'article_content' field MISSING from JSON output.")
            
    except Exception as e:
        print(f"❌ Error invoking tool: {e}")

if __name__ == "__main__":
    asyncio.run(verify_tool())
