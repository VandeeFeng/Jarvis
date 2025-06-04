import asyncio
from db.database import init_db, SessionLocal
from config.main import search_memories

async def test_memory_retrieval():
    # Initialize database
    init_db()
    db = SessionLocal()
    
    try:
        # Test different keywords including the one we just stored
        keywords_to_test = ["readability", "python", "programming"]
        
        for keyword in keywords_to_test:
            print(f"\nSearching for memories with keyword: {keyword}")
            retrieved_memories = await search_memories(keyword=keyword, db=db)
            
            if retrieved_memories:
                print(f"Found {len(retrieved_memories)} memories:")
                for mem in retrieved_memories:
                    print(f"\nID: {mem.id}")
                    print(f"Content: {mem.content}")
                    print(f"Keyword: {mem.keyword}")
                    print(f"Created at: {mem.created_at}")
            else:
                print(f"No memories found with keyword: {keyword}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_memory_retrieval()) 