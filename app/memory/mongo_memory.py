from typing import Dict, List, Optional

from bson import ObjectId

from app.memory.base import MemoriaCtx, MemoryStore
from app.memory.resumo_service import ResumoService
from app.repository.postgresql.perfil_repository import PerfilContext


class MongoMemory(MemoryStore):
    def __init__(self, repository, resumo_service: ResumoService, janela_mensagens: int = 10):
        super().__init__(repository, janela_mensagens)
        self.resumo_service = resumo_service

    def preparar_turno(
        self,
        pergunta: str,
        perfil_ctx: PerfilContext,
        session_id: str,
        usuario_id: Optional[str] = None,
    ) -> MemoriaCtx:
        """
        Chamado pelo Guardrail de entrada antes do roteamento: garante que
        existe sessão ativa (criando se necessário) e registra a pergunta
        do usuário. Devolve o contexto que o resto do grafo pode precisar.
        """
        sessao = self.repository.buscar_sessao_ativa(session_id)
        if sessao is None:
            doc_id = self.repository.criar_sessao(session_id, perfil_ctx.perfil_id, usuario_id=usuario_id)
        else:
            doc_id = ObjectId(sessao.id)

        self.repository.adicionar_mensagem(doc_id, role="usuario", content=pergunta)

        sessao_atualizada = self.repository.buscar_por_id(doc_id)
        if sessao_atualizada is None:
            return MemoriaCtx()

        return MemoriaCtx(
            resumo=sessao_atualizada.resumo,
            mensagens_recentes=sessao_atualizada.mensagens[-self.janela_mensagens:],
            agentes_chamados=sessao_atualizada.agentes_chamados,
        )

    def registrar_resposta(
        self,
        resposta: str,
        session_id: str,
        agentes: Optional[List[str]] = [],
        meta: Optional[Dict] = None,
    ) -> MemoriaCtx:
        """Chamado pelo Orquestrador ao final do turno."""
        sessao = self.repository.buscar_sessao_ativa(session_id)
        if sessao is None:
            return MemoriaCtx()

        doc_id = ObjectId(sessao.id)
        self.repository.adicionar_mensagem(doc_id, role="assistente", content=resposta, agentes=agentes, meta=meta)

        sessao_atualizada = self.repository.buscar_por_id(doc_id)
        if sessao_atualizada is None:
            return MemoriaCtx()
        
        return MemoriaCtx(
            resumo=sessao_atualizada.resumo,
            mensagens_recentes=sessao_atualizada.mensagens[-self.janela_mensagens:],
            agentes_chamados=sessao_atualizada.agentes_chamados,
        )

    def encerrar_sessao(self, session_id: str) -> MemoriaCtx:
        """
        Decide se vale encerrar a sessão (só encerra se ela teve pelo menos
        uma mensagem) e, quando um resumo for fornecido, persiste junto.
        """
        sessao = self.repository.buscar_sessao_ativa(session_id)
        if sessao is None or not sessao.mensagens:
            return MemoriaCtx()

        doc_id = ObjectId(sessao.id)

        resumo = ''
        resumo = self.resumo_service.gerar_resumo(sessao.mensagens, sessao.resumo)
        self.repository.marcar_encerrada(ObjectId(sessao.id), resumo=resumo)

        sessao_atualizada = self.repository.buscar_por_id(doc_id)
        if sessao_atualizada is None:
            return MemoriaCtx()
        
        return MemoriaCtx(
            resumo=sessao_atualizada.resumo,
            mensagens_recentes=sessao_atualizada.mensagens[-self.janela_mensagens:],
            agentes_chamados=sessao_atualizada.agentes_chamados,
        )

    def recuperar_historico(self, perfil_ctx: PerfilContext, user_id: Optional[str]):
        filtro = {
            'usuario_id' if user_id else 'perfil_id': (user_id or perfil_ctx.perfil_id),
            'resumo': {'$nin': ['', None]}
        }

        sessoes = self.repository.buscar_usuario(filtro)
        return [
            {'doc_id': sessao.id, 'data_inicio': sessao.data_inicio, 'resumo': sessao.resumo}
            for sessao in sessoes
        ]