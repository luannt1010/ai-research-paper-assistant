CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks_db (
    id BIGSERIAL PRIMARY KEY,
    document_id VARCHAR(255),
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);