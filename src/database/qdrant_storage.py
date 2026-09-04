from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from .base import BaseVectorStore
from .settings import Settings

class QdrantVectorStore(BaseVectorStore):
    def __init__(self):

        settings = Settings()
        self.url = settings.qdrant_url
        self.api_key = settings.qdrant_api_key
        self.collection_name = settings.qdrant_collection_name
        self.client = self._get_connection()

    def _get_connection(self):
        return QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=600
        )

    def _build_filter(self, filters: Optional[Dict[str, Any]]=None):
        if filters is None:
            return None
        if not filters:
            raise ValueError( "Filters cannot be empty")

        conditions = []
        for key, value in filters.items():
            # document_id nằm trực tiếp trong payload
            if key == "document_id":
                payload_key = "document_id"
            else:
                payload_key = f"metadata.{key}"

            conditions.append(
                models.FieldCondition(
                    key=payload_key,
                    match=models.MatchValue(
                        value=value
                    )
                )
            )

        return models.Filter(must=conditions)

    def insert(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        points = []
        for i, (chunk, embedding) in enumerate(zip(documents, embeddings), start=1):
            point = models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "document_id": chunk.metadata.get("document_id"),
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        query_filter = self._build_filter(filters)

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        results = []
        for point in result.points:
            payload = point.payload
            results.append({
                "id": point.id,
                "document_id": payload.get("document_id"),
                "content": payload.get("content"),
                "metadata": payload.get("metadata"),
                "similarity": point.score
            })
        return results

    def delete(self, filters: Optional[Dict[str, Any]]=None) -> None:
        delete_filter = self._build_filter(filters)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=delete_filter),
            wait=True
        )

    def count(self, filters: Optional[Dict[str, Any]]=None) -> int:
        count_filter = self._build_filter(filters)
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True
        )
        return result.count

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        all_chunks = []
        next_page_offset = None
        while True:
            response, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_vectors=False,
                with_payload=True,
                offset=next_page_offset
            )
            all_chunks.extend(response)
            if next_page_offset is None:
                break
        results = []
        for point in all_chunks:
            payload = point.payload
            results.append({
                "id": point.id,
                "document_id": payload.get("document_id"),
                "content": payload.get("content"),
                "metadata": payload.get("metadata"),
                "similarity": point.score
            })
        return results

    def close(self) -> None:
        self.client.close()
