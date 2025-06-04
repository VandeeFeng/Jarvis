-- Add user_id column if it doesn't exist
DO $$ 
BEGIN 
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='memories' AND column_name='user_id'
    ) THEN
        ALTER TABLE memories ADD COLUMN user_id TEXT;
    END IF;
END $$;

-- Add index for user_id if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE tablename = 'memories' AND indexname = 'idx_memories_user_id'
    ) THEN
        CREATE INDEX idx_memories_user_id ON memories(user_id);
    END IF;
END $$;

-- Add compound index for user_id + created_at if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE tablename = 'memories' AND indexname = 'idx_memories_user_created'
    ) THEN
        CREATE INDEX idx_memories_user_created ON memories(user_id, created_at DESC);
    END IF;
END $$; 