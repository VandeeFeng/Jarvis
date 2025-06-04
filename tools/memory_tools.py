from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from db.database import SessionLocal
import numpy as np
from config.config import MEMORY

def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Convert to numpy arrays for efficient calculation
    a = np.array(embedding1)
    b = np.array(embedding2)
    # Calculate cosine similarity
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def format_memory_context(user_memories: List, general_memories: List) -> str:
    """Format memory context string from user-specific and general memories with improved deduplication."""
    # Format user-specific memories with deduplication
    user_specific_context_str = ""
    if user_memories:
        # Track unique contents both by exact match and similarity
        seen_contents = set()
        unique_memories = []
        
        for mem in user_memories:
            normalized_content = mem.content.lower().strip()
            # Check if we've seen this exact content
            if normalized_content not in seen_contents:
                seen_contents.add(normalized_content)
                # Also check for similar content using embeddings
                is_similar = False
                for existing_mem in unique_memories:
                    similarity = calculate_cosine_similarity(mem.embedding, existing_mem.embedding)
                    if similarity > MEMORY["similarity_threshold"]:
                        is_similar = True
                        break
                if not is_similar:
                    unique_memories.append(mem)
        
        if unique_memories:
            user_specific_context_str = "\n\nRelevant personal memories for you:\n"
            for mem in unique_memories:
                user_specific_context_str += f"- {mem.content} (Keyword: {mem.keyword})\n"

    # Format general memories with improved deduplication
    general_memories_context_str = ""
    if general_memories:
        # Get embeddings of all user memories for similarity comparison
        user_embeddings = [(mem.embedding, mem.content.lower().strip()) 
                          for mem in user_memories] if user_memories else []
        
        filtered_memories = []
        for mem in general_memories:
            normalized_content = mem.content.lower().strip()
            # Skip if exact match exists in user memories
            if any(content == normalized_content for _, content in user_embeddings):
                continue
                
            # Check for similarity with user memories
            is_similar = False
            for embedding, _ in user_embeddings:
                similarity = calculate_cosine_similarity(mem.embedding, embedding)
                if similarity > MEMORY["similarity_threshold"]:
                    is_similar = True
                    break
                    
            if not is_similar:
                # Also check similarity with already filtered memories
                for filtered_mem in filtered_memories:
                    similarity = calculate_cosine_similarity(mem.embedding, filtered_mem.embedding)
                    if similarity > MEMORY["similarity_threshold"]:
                        is_similar = True
                        break
                        
            if not is_similar:
                filtered_memories.append(mem)
        
        if filtered_memories:
            general_memories_context_str = "\n\nOther relevant information (from general semantic search):\n"
            for mem in filtered_memories:
                general_memories_context_str += f"- {mem.content}\n"

    # Combine memory contexts
    combined_memory_context = user_specific_context_str + general_memories_context_str
    if not combined_memory_context.strip():  # If both are empty
        combined_memory_context = "\n\nRelevant memories:\nNo specific memories found for this query, and no personal memories loaded."

    return combined_memory_context

async def process_memory_response(memory: Optional[str], user_id: str, db: Session) -> Tuple[str, Optional[str]]:
    """Process the memory response and store if valid with deduplication."""
    if memory and ': ' in memory:
        # Split the memory into category and content
        category, fact = memory.split(': ', 1)
        
        # Check if this is a valid category
        valid_categories = {'preference', 'purchase', 'location', 'schedule', 'contact', 'personal'}
        if category.lower() not in valid_categories:
            return "", None
            
        # Store in database using the category as keyword, with deduplication
        from config.main import store_memory_in_db, get_embedding, search_similar_memories
        
        # Get embedding for the new content
        new_embedding = await get_embedding(fact)
        
        # Search for similar existing memories
        similar_memories = await search_similar_memories(
            query=fact,
            db=db,
            limit=5,
            ef_search=MEMORY["ef_search"]
        )
        
        # Check for similarities
        for mem in similar_memories:
            similarity = calculate_cosine_similarity(new_embedding, mem.embedding)
            if similarity > MEMORY["similarity_threshold"]:
                # If very similar content exists, don't store new memory
                return "", None
        
        # If no similar memory found, store the new one
        await store_memory_in_db(
            content=fact,
            keyword=category,
            db=db,
            user_id=user_id
        )
        return fact, category
        
    return "", None 