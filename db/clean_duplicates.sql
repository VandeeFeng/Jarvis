-- HOW TO USE THIS SCRIPT:
-- Method 1 (Direct):
--   docker exec -i $(docker ps -qf "ancestor=pgvector/pgvector:pg17") psql -U postgres -d memory_store -f /docker-entrypoint-initdb.d/clean_duplicates.sql
--
-- Method 2 (Using psql if installed):
--   PGPASSWORD=postgres psql -h localhost -U postgres -d memory_store -f clean_duplicates.sql
--
-- This script will:
-- 1. Find duplicate memories using both content similarity and exact matching
-- 2. Keep the oldest record of each unique content
-- 3. Delete all duplicates
-- 4. Vacuum the table to reclaim storage
-- 5. (Optional) Show remaining records

-- Clean duplicate memories while preserving the oldest record of each unique content
-- This script uses cosine similarity for vector comparison and exact text matching

-- First, create a view to identify similar pairs
WITH SimilarPairs AS (
  SELECT 
    a.id as id1,
    b.id as id2,
    a.content as content1,
    b.content as content2,
    a.keyword as keyword1,
    b.keyword as keyword2,
    (1 - (a.embedding <=> b.embedding)) as similarity,
    a.created_at as created_at1,
    b.created_at as created_at2
  FROM memories a
  JOIN memories b ON a.id < b.id
  WHERE 
    -- Match by vector similarity (95% or higher)
    (1 - (a.embedding <=> b.embedding)) > 0.95
    -- OR by exact content match (case-insensitive, trimmed)
    OR LOWER(TRIM(a.content)) = LOWER(TRIM(b.content))
),

-- Select records to keep (earliest record for each unique content)
KeepRecords AS (
  SELECT MIN(id) as keep_id 
  FROM memories 
  GROUP BY LOWER(TRIM(content))
)

-- Delete all records except the ones we want to keep
DELETE FROM memories 
WHERE id NOT IN (SELECT keep_id FROM KeepRecords)
RETURNING id, content, keyword, created_at;

-- Vacuum the table to reclaim storage and update statistics
VACUUM ANALYZE memories;

-- Optional: Show remaining records
-- SELECT id, content, keyword, created_at 
-- FROM memories 
-- ORDER BY id; 