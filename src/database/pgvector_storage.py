import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from typing import List, Optional, Dict, Any
from psycopg.types.json import Jsonb
from langchain_core.documents import Document
from .base import BaseVectorStore
from .settings import Settings


class PGVectorStore(BaseVectorStore):
    def __init__(self):

        settings = Settings()
        self.db_url = settings.pgvector_db_url
        self.conn = self._get_connection()

    def _get_connection(self) -> psycopg.Connection:
        connection = psycopg.connect(self.db_url, row_factory=dict_row)
        register_vector(connection)
        return connection

    def _parse_filters(self, filters: Optional[Dict[str, Any]]=None):
        if filters is None:
            return "", []
        if len(filters) == 0:
            raise ValueError("Filters cannot be empty")
        conditions = []
        params = []
        for key, value in filters.items():

            if key == "document_id":
                conditions.append("document_id = %s")
                params.append(value)
            else:
                # Các field khác nằm trong metadata JSONB
                conditions.append(f"metadata->>%s = %s")
                params.extend([key, str(value)])

        where_clause = (
                "WHERE " + " AND ".join(conditions)
        )
        return where_clause, params

    def insert(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        insert_query = """
        INSERT INTO chunks_db (document_id, content, metadata, embedding)
        VALUES (%s, %s, %s, %s)
        """

        rows = []
        for chunk, embedding in zip(documents, embeddings):
            rows.append((chunk.metadata.get("document_id"),
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

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        where_clause, filter_params = self._parse_filters(filters)
        select_query = f"""
        SELECT id, document_id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
        FROM chunks_db
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        params = [query_embedding, *filter_params, query_embedding, top_k]
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(select_query, params)
                return cursor.fetchall()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def delete(self, filters: Optional[Dict[str, Any]] = None) -> None:
        where_clause, filter_params = self._parse_filters(filters)
        query = f"""
        DELETE FROM chunks_db
        {where_clause}
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, filter_params)
            self.conn.commit()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        where_clause, filter_params = self._parse_filters(filters)
        query = f"""
        SELECT COUNT(*) AS total 
        FROM chunks_db
        {where_clause}
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, filter_params)
                result = cursor.fetchone()
            return result["total"]
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        query = """
        SELECT id, document_id, content, metadata
        FROM chunks_db
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()
        except psycopg.Error as e:
            print(e)
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()


