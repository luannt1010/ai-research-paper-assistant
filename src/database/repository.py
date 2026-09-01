import psycopg
from typing import List
from psycopg.types.json import Jsonb
from .connect_db import get_connection
from langchain_core.documents import Document

class Repository:
    def __init__(self, benchmark: bool = False):
        self.conn = get_connection()
        self.benchmark = benchmark

    def insert_chunks(self, chunks: list[Document], embeddings: List[List[float]]):
        insert_query = """INSERT INTO chunks_db (document_id, content, metadata, embedding, benchmark)
        VALUES (%s, %s, %s, %s, %s)"""

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append((chunk.metadata.get("document_id"),
                         chunk.page_content,
                         Jsonb(chunk.metadata),
                         embedding,
                         self.benchmark))
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
        WHERE benchmark = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(select_query, (query_embedding, self.benchmark, query_embedding, top_k))
                return cursor.fetchall()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def get_all_chunks(self):
        query = """SELECT id, document_id, content, metadata
        FROM chunks_db
        WHERE benchmark = %s;
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (self.benchmark,))
                return cursor.fetchall()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def check_benchmark(self):
        query = """SELECT count(*)
        FROM chunks_db
        WHERE benchmark = TRUE;"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchone()["count"]
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def delete_benchmark(self):
        query = """
            DELETE FROM chunks_db
            WHERE benchmark = TRUE;
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
            self.conn.commit()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise
    
    def close(self):
        self.conn.close()


if __name__ == "__main__":
    repo = Repository(True)
    res = repo.check_benchmark()
    print(res)
