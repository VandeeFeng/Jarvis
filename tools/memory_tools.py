from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.orm import Session
from db.database import SessionLocal
import numpy as np
from config.config import MEMORY, VALID_CATEGORIES

class MemoryTools:
    """Tools for managing and processing memory-related operations."""
    
    def __init__(self):
        self.db = None
        self.user_id = None
        
    @property
    def function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions for memory tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memories_by_keyword",
                    "description": "Search for memories using a specific keyword or category.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "The keyword or category to search for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of memories to return",
                                "default": 5
                            }
                        },
                        "required": ["keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memories_by_similarity",
                    "description": "Search for memories that are semantically similar to a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query text to find similar memories for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of memories to return",
                                "default": 5
                            },
                            "ef_search": {
                                "type": "integer",
                                "description": "Size of the dynamic candidate list for similarity search",
                                "default": 100
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_memory",
                    "description": "Store a new memory with a specific category.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to store"
                            },
                            "category": {
                                "type": "string",
                                "description": "The category of the memory",
                                "enum": list(VALID_CATEGORIES)
                            }
                        },
                        "required": ["content", "category"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_memory_context",
                    "description": "Get formatted memory context based on a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to find relevant memories for"
                            },
                            "include_user_memories": {
                                "type": "boolean",
                                "description": "Whether to include user-specific memories",
                                "default": True
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def set_db(self, db: Session):
        """Set the database session"""
        self.db = db

    def set_user_id(self, user_id: str):
        """Set the current user ID"""
        self.user_id = user_id

    @staticmethod
    def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def search_memories_by_keyword(self, keyword: str, limit: int = 5) -> Dict[str, Any]:
        """Search for memories using a specific keyword."""
        from config.main import search_memories
        
        memories = await search_memories(
            keyword=keyword,
            db=self.db,
            limit=limit
        )
        
        return {
            "memories": [
                {
                    "content": mem.content,
                    "keyword": mem.keyword,
                    "created_at": mem.created_at.isoformat()
                }
                for mem in memories
            ]
        }

    async def search_memories_by_similarity(self, query: str, limit: int = 5, ef_search: int = 100) -> Dict[str, Any]:
        """Search for memories that are semantically similar to a query."""
        from config.main import search_similar_memories
        from config.config import MEMORY
        
        memories = await search_similar_memories(
            query=query,
            db=self.db,
            limit=limit,
            ef_search=ef_search,
            user_id=self.user_id,
            similarity_threshold=MEMORY["similarity_threshold"]
        )
        
        return {
            "memories": [
                {
                    "content": mem.content,
                    "keyword": mem.keyword,
                    "similarity_score": mem.similarity if hasattr(mem, 'similarity') else None,
                    "created_at": mem.created_at.isoformat()
                }
                for mem in memories
            ]
        }

    async def store_memory(self, content: str, category: str) -> Dict[str, Any]:
        """Store a new memory with the specified category."""
        if category.lower() not in VALID_CATEGORIES:
            return {
                "success": False,
                "error": f"Invalid category. Must be one of: {', '.join(VALID_CATEGORIES)}"
            }

        try:
            from config.main import store_memory_in_db, get_embedding
            
            # Get embedding for the new content
            embedding = await get_embedding(content)
            
            # Check for similar existing memories to avoid duplicates
            similar_memories = await self.search_memories_by_similarity(
                query=content,
                limit=1
            )
            
            if similar_memories.get("memories"):
                first_memory = similar_memories["memories"][0]
                similarity = self.calculate_cosine_similarity(
                    embedding,
                    await get_embedding(first_memory["content"])
                )
                
                if similarity > MEMORY["similarity_threshold"]:
                    return {
                        "success": False,
                        "error": "Similar memory already exists",
                        "existing_memory": first_memory
                    }
            
            # Store the new memory
            stored_memory = await store_memory_in_db(
                content=content,
                keyword=category.lower(),
                db=self.db,
                user_id=self.user_id
            )
            
            return {
                "success": True,
                "stored_memory": {
                    "content": stored_memory.content,
                    "category": stored_memory.keyword,
                    "created_at": stored_memory.created_at.isoformat()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def get_memory_context(
        self,
        query: str,
        include_user_memories: bool = True,
        include_general_memories: bool = True
    ) -> Dict[str, Any]:
        """Get formatted memory context based on a query."""
        user_memories = []
        general_memories = []
        
        if include_user_memories:
            # Check if query is a valid category
            if query.lower() in VALID_CATEGORIES:
                # Use exact keyword matching for categories
                user_result = await self.search_memories_by_keyword(keyword=query.lower())
            else:
                # Use semantic search for other queries
                user_result = await self.search_memories_by_similarity(
                    query=query,
                    limit=MEMORY["similar_memory_limit"],
                    ef_search=MEMORY["ef_search"]
                )
            user_memories = user_result.get("memories", [])
            
        if include_general_memories:
            # Implement general memory search logic here
            pass
            
        formatted_context = await self.format_memory_context(user_memories, general_memories)
        
        return {
            "formatted_context": formatted_context,
            "user_memories": user_memories,
            "general_memories": general_memories
        }

    @staticmethod
    async def format_memory_context(user_memories: List[Dict], general_memories: List[Dict]) -> str:
        """Format memory context string from user-specific and general memories."""
        context_parts = []
        
        if user_memories:
            context_parts.append("\nRelevant personal memories for you:")
            for mem in user_memories:
                context_parts.append(f"- {mem['content']} (Keyword: {mem['keyword']})")
                
        if general_memories:
            context_parts.append("\nOther relevant information (from general semantic search):")
            for mem in general_memories:
                context_parts.append(f"- {mem['content']}")
                
        if not context_parts:
            return "\nNo relevant memories found for this query."
            
        return "\n".join(context_parts) 