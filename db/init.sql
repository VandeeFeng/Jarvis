CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the existing table if it exists
DROP TABLE IF EXISTS memories;

-- Create the table with the correct vector dimension (768 for nomic-embed-text)
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    keyword TEXT NOT NULL,
    user_id TEXT
);

-- Create HNSW index for fast similarity search
CREATE INDEX hnsw_embedding_idx ON memories 
USING hnsw(embedding vector_l2_ops)
WITH (
    m = 16,        -- max number of connections per layer (default: 16)
    ef_construction = 64  -- size of the dynamic candidate list for construction (default: 64)
);

-- Create index for fast user_id lookups
CREATE INDEX idx_memories_user_id ON memories(user_id);

-- Create index for timestamp-based sorting within user_id groups
CREATE INDEX idx_memories_user_created ON memories(user_id, created_at DESC); 