from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
import os
from dotenv import load_dotenv
from db.database import get_db, Memory, init_db
from typing import List, Dict
import numpy as np
from ollama import AsyncClient
from config.config import EMBEDDING
import logging

load_dotenv()

app = FastAPI()
ollama_client = AsyncClient()

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:14b"  # or another model you have pulled

logger = logging.getLogger(__name__)

# Define the function for OpenAI to call
functions = [
    {
        "name": "store_memory",
        "description": "Store information in the memory database",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to store"
                },
                "keyword": {
                    "type": "string",
                    "description": "The keyword associated with the content"
                }
            },
            "required": ["content", "keyword"]
        }
    }
]

async def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama with error handling."""
    try:
        # Using Ollama's embedding endpoint with nomic-embed-text
        response = await ollama_client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        
        # Check if response has the expected structure
        if not hasattr(response, 'embeddings') and not hasattr(response, 'embedding'):
            raise ValueError("Unexpected response format from Ollama")
            
        # Get embedding from response (handle both possible response formats)
        embedding = response.embeddings[0] if hasattr(response, 'embeddings') else response.embedding
        
        # Validate embedding
        if not embedding or not isinstance(embedding, list):
            raise ValueError("Invalid embedding format received")
            
        if len(embedding) != EMBEDDING["dimensions"]:
            raise ValueError(f"Expected {EMBEDDING['dimensions']} dimensions, got {len(embedding)}")
            
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

async def store_memory_in_db(content: str, keyword: str, db: Session, user_id: str = None):
    embedding = await get_embedding(content)
    memory = Memory(
        content=content,
        embedding=embedding,
        keyword=keyword,
        user_id=user_id
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def search_similar_memories(query: str, db: Session, limit: int = 3, ef_search: int = 100, user_id: str = None) -> List[Memory]:
    # Get embedding for the query
    query_embedding = await get_embedding(query)
    
    # Use HNSW index for fast similarity search
    memories = Memory.find_similar(
        db=db,
        query_embedding=query_embedding,
        limit=limit,
        ef_search=ef_search,
        user_id=user_id
    )
    
    return memories

async def get_user_specific_memories(user_id: str, db: Session, limit: int = 5) -> List[Memory]:
    if not user_id:
        return []
    return db.query(Memory).filter(Memory.user_id == user_id).order_by(desc(Memory.created_at)).limit(limit).all()

@app.post("/process")
async def process_input(user_input: str, db: Session = Depends(get_db)):
    # Get completion from Ollama
    response = await ollama_client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    
    # Store the response
    content = response.message.content
    keyword = "chat_response"
    
    # Store in database
    memory = await store_memory_in_db(content, keyword, db)
    return {"status": "success", "stored_memory": memory}

@app.get("/search")
async def search_memories(keyword: str, db: Session = Depends(get_db)):
    memories = db.query(Memory).filter(Memory.keyword == keyword).all()
    return memories

if __name__ == "__main__":
    init_db()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 