from ast import MatchValue
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.grpc import FieldCondition, Filter, PointStruct

from app.repository.base import Repository


class ResumoPayload(BaseModel):
    usuario_id:  str
    perfil_id:   str
    sessao_id:   str
    resumo:      str
    iniciada_em: datetime

class ResultadoBusca[T](BaseModel):
    item: T
    score: float

class ResumoRepository(Repository[ResumoPayload, QdrantClient]):
    COLLECTION_MEMORIA = "memoria_resumos"

    def __init__(self, db):
        super().__init__(db)

    def upsert(self, vetor, payload: ResumoPayload):
        self.client.upsert(self.COLLECTION_MEMORIA, points=[PointStruct(
            id=str(uuid4(payload.sessao_id)),
            vector=vetor,
            payload=payload.model_dump(mode="json"),
        )])

    def buscar(self, texto: str, usuario_id: str, k: int = 5) -> list[ResultadoBusca[ResumoPayload]]:
        hits = self.client.query_points(
            self.collection,
            query=self.embedder.embed_query(texto),
            limit=k,
            query_filter=Filter(must=[FieldCondition(key="usuario_id", match=MatchValue(value=usuario_id))]),
        ).points

        return [
            ResultadoBusca(item=ResumoPayload.model_validate(h.payload), score=h.score)
            for h in hits
        ]