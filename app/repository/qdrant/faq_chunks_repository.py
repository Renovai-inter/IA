from typing import List

from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.repository.base import Repository


class FaqChunksPayload(BaseModel):
    page_content: str
    page_number: int
    source: str

class ResultadoBusca[T](BaseModel):
    id: str
    score: float
    item: T

class FaqChunksRepository(Repository[FaqChunksPayload, QdrantClient]):
    def __init__(self, db: QdrantClient, collection: str):
        super().__init__(db)
        self._collection = collection


    def buscar_faq(self, embedding, k = 6) -> List[ResultadoBusca]:
        resultados = self._db.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=k,
        )

        if not resultados.points:
            return []

        return [
            ResultadoBusca(id=str(p.id), score=p.score, item=FaqChunksPayload.model_validate(p.payload))
            for p in resultados.points
        ]