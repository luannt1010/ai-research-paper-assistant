import psycopg
from typing import List
from psycopg.types.json import Jsonb
from .connect_db import get_connection
from langchain_core.documents import Document

class Repository:
    def __init__(self):
        self.conn = get_connection()

    def insert_chunks(self, chunks: list[Document], embeddings: List[List[float]]):
        insert_query = """INSERT INTO chunks_db (document_id, content, metadata, embedding)
        VALUES (%s, %s, %s, %s)"""

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append((chunk.metadata.get("source"),
                         chunk.page_content,
                         Jsonb(chunk.metadata),
                         embedding))
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(insert_query, rows)
            self.conn.commit()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def similarity_search(self, query_embedding: List[float], top_k: int = 5):
        select_query = """SELECT id, document_id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
        FROM chunks_db
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(select_query, (query_embedding, query_embedding, top_k))
                return cursor.fetchall()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def get_all_chunks(self):
        query = """SELECT id, document_id, content, metadata
        FROM chunks_db;
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def close(self):
        self.conn.close()

