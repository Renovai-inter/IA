from typing import List, Optional

from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.repository.base import Repository


class FaqChunksPayload(BaseModel):
    page_content: str
    page_number: int = 0
    source: str = ''
    artigo: Optional[str] = None
    titulo: Optional[str] = None
    capitulo: Optional[str] = None


class ResultadoBusca[T](BaseModel):
    id: str
    score: float
    item: T


class FaqChunksRepository(Repository[FaqChunksPayload, QdrantClient]):
    def __init__(self, db: QdrantClient, collection: str):
        super().__init__(db)
        self._collection = collection

    @property
    def collection(self) -> str:
        return self._collection

    def buscar(self, embedding, k: int = 6) -> List[ResultadoBusca[FaqChunksPayload]]:
        try:
            resultados = self._db.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=k,
            )
        except Exception:
            return []

        if not resultados.points:
            return []

        return [
            ResultadoBusca(id=str(p.id), score=p.score, item=FaqChunksPayload.model_validate(p.payload))
            for p in resultados.points
        ]

    def buscar_faq(self, embedding, k: int = 6) -> List[ResultadoBusca[FaqChunksPayload]]:
        return self.buscar(embedding=embedding, k=k)
