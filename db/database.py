from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
from sqlalchemy.types import UserDefinedType
import numpy as np
from typing import List

# Define expected embedding dimension for nomic-embed-text
EMBEDDING_DIM = 768  # nomic-embed-text outputs 768-dimensional vectors

# Define a custom type for pgvector
class Vector(UserDefinedType):
    def get_col_spec(self):
        return "vector"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, (list, np.ndarray)):
                return f"[{','.join(map(str, value))}]"
            return value
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            # Convert string representation back to list if needed
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                return [float(x) for x in value[1:-1].split(',')]
            return value
        return process

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/memory_store"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Memory(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    embedding = Column(Vector, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    keyword = Column(String, nullable=False)
    user_id = Column(String, nullable=True, index=True)

    def __init__(self, *args, **kwargs):
        if 'embedding' in kwargs:
            embedding = kwargs['embedding']
            if len(embedding) != EMBEDDING_DIM:
                raise ValueError(f"Embedding must be exactly {EMBEDDING_DIM} dimensions, got {len(embedding)}")
        self.user_id = kwargs.pop('user_id', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def find_similar(cls, db, query_embedding: List[float], limit: int = 5, ef_search: int = 100):
        """
        Find similar memories using HNSW index
        :param db: Database session
        :param query_embedding: Query vector
        :param limit: Number of results to return
        :param ef_search: Size of the dynamic candidate list for search (higher = more accurate but slower)
        :return: List of Memory objects sorted by similarity
        """
        # Set search parameters for HNSW
        db.execute(text(f"SET hnsw.ef_search = {ef_search}"))
        
        # Convert the embedding list to a PG vector string format
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        # Use vector_l2_ops operator <-> for L2 distance (lower = more similar)
        sql = text("""
            SELECT id, content, keyword, created_at, embedding::text
            FROM memories
            ORDER BY embedding <-> CAST(:query_embedding AS vector)
            LIMIT :limit
        """)
        
        result = db.execute(
            sql,
            {
                "query_embedding": vector_str,
                "limit": limit
            }
        )
        
        # Convert results to Memory objects
        memories = []
        for row in result:
            # Convert the embedding string back to a list
            embedding_str = row.embedding
            embedding = [float(x) for x in embedding_str[1:-1].split(',')]
            
            memory = Memory(
                id=row.id,
                content=row.content,
                keyword=row.keyword,
                created_at=row.created_at,
                embedding=embedding,
                user_id=getattr(row, 'user_id', None)
            )
            memories.append(memory)
        
        return memories

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 