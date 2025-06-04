from typing import Optional, List
from sqlalchemy.orm import Session
from db.database import SessionLocal

async def format_memory_context(user_memories: List, general_memories: List) -> str:
    """Format memory context string from user-specific and general memories."""
    # Format user-specific memories
    user_specific_context_str = ""
    if user_memories:
        user_specific_context_str = "\n\nRelevant personal memories for you:\n"
        for mem in user_memories:
            user_specific_context_str += f"- {mem.content} (Keyword: {mem.keyword})\n"

    # Format general memories
    general_memories_context_str = ""
    if general_memories:
        # Filter out memories that are already in user-specific memories
        filtered_memories = [
            mem for mem in general_memories 
            if not any(user_mem.content == mem.content 
                      for user_mem in (user_memories or []))
        ]
        
        if filtered_memories:
            general_memories_context_str = "\n\nOther relevant information (from general semantic search):\n"
            for mem in filtered_memories:
                general_memories_context_str += f"- {mem.content}\n"

    # Combine memory contexts
    combined_memory_context = user_specific_context_str + general_memories_context_str
    if not combined_memory_context.strip():  # If both are empty
        combined_memory_context = "\n\nRelevant memories:\nNo specific memories found for this query, and no personal memories loaded."

    return combined_memory_context

async def process_memory_response(memory: Optional[str], user_id: str, db: Session) -> tuple[str, Optional[str]]:
    """Process the memory response and store if valid."""
    if memory and ': ' in memory:
        # Split the memory into category and content
        category, fact = memory.split(': ', 1)
        
        # Store in database using the category as keyword
        from config.main import store_memory_in_db
        await store_memory_in_db(
            content=fact,  # Store only the fact part
            keyword=category,  # Use the category as keyword
            db=db,
            user_id=user_id
        )
        return fact, category
    return "", None 