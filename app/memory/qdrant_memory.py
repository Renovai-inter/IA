from datetime import datetime
from typing import Optional

from qdrant_client.grpc import FieldCondition, Filter
from qdrant_client.models import MatchValue

from app.memory.base import MemoriaCtx, MemoryStore
from app.llms.resumo_service import ResumoService
from app.repository.qdrant.resumo_repository import ResumoPayload


class QdrantMemory(MemoryStore):
    def __init__(self, repository, resumo_service: ResumoService, janela = 5):
        super().__init__(repository, resumo_service, janela)

    def atualizar_memoria(self, memory_ctx: MemoriaCtx):
        vetor = self.resumo_service.gerar_embedding(memory_ctx.resumo)
        
        payload = ResumoPayload(
            usuario_id=memory_ctx.usuario_id,
            perfil_id=memory_ctx.perfil_id,
            sessao_id=memory_ctx.sessao_id,
            resumo=memory_ctx.resumo,
            data_inicio=memory_ctx.data_inicio
        )
        self.repository.upsert(vetor, payload)

    def recuperar_historico(self, user_id: Optional[str], perfil_id: str, busca: str):
        embedding = self.resumo_service.gerar_embedding(busca)
        
        filtro = Filter(
            must=[FieldCondition(
                key='usuario_id' if user_id else 'perfil_id',
                match=MatchValue(value=(user_id or perfil_id))
            )]
        )

        resultados = self.repository.buscar(embedding, filtro, self.janela)
        return [
            MemoriaCtx(
                doc_id=resultado.id,
                usuario_id=resultado.item.usuario_id,
                perfil_id=resultado.item.perfil_id,
                sessao_id=resultado.item.sessao_id,
                resumo=resultado.item.resumo,
                data_inicio=resultado.item.data_inicio,
            )
            for resultado in resultados
        ]