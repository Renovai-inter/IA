from ast import MatchValue
from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.grpc import Filter, PointStruct

from app.repository.base import Repository


class ResumoPayload(BaseModel):
    usuario_id:  Optional[str]
    perfil_id:   str
    sessao_id:   str
    resumo:      str
    data_inicio: datetime

class ResultadoBusca[T](BaseModel):
    id: str
    score: float
    item: T

class ResumoRepository(Repository[ResumoPayload, QdrantClient]):
    COLLECTION_MEMORIA = "memoria_resumos"

    def __init__(self, db: QdrantClient):
        super().__init__(db)

    def upsert(self, vetor, payload: ResumoPayload):
        self._db.upsert(self.COLLECTION_MEMORIA, points=[PointStruct(
            id=str(uuid4()),
            vector=vetor,
            payload=payload.model_dump(mode="json"),
        )])

    def buscar(self, embedding, filter: Filter, k: int = 5) -> list[ResultadoBusca[ResumoPayload]]:
        hits = self._db.query_points(
            self.COLLECTION_MEMORIA,
            query=embedding,
            limit=k,
            query_filter=filter,
        ).points

        return [
            ResultadoBusca(id=str(h.id), score=h.score, item=ResumoPayload.model_validate(h.payload))
            for h in hits
        ]