from typing import Dict, List, Optional

from bson import ObjectId

from app.memory.base import MemoriaCtx, MemoryStore
from app.llms.resumo_service import ResumoService
from app.repository.postgresql.perfil_repository import PerfilContext


class MongoMemory(MemoryStore):
    def __init__(self, repository, resumo_service: ResumoService, janela: int = 10):
        super().__init__(repository, resumo_service, janela)

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
            doc_id=sessao_atualizada.id,
            usuario_id=sessao_atualizada.usuario_id,
            perfil_id=sessao_atualizada.perfil_id,
            sessao_id=sessao_atualizada.session_id,
            resumo=sessao_atualizada.resumo,
            data_inicio=sessao_atualizada.data_inicio,
            mensagens_recentes=sessao_atualizada.mensagens[-self.janela:],
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
            doc_id=sessao_atualizada.id,
            usuario_id=sessao_atualizada.usuario_id,
            perfil_id=sessao_atualizada.perfil_id,
            sessao_id=sessao_atualizada.session_id,
            resumo=sessao_atualizada.resumo,
            data_inicio=sessao_atualizada.data_inicio,
            mensagens_recentes=sessao_atualizada.mensagens[-self.janela:],
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
            doc_id=sessao_atualizada.id,
            usuario_id=sessao_atualizada.usuario_id,
            perfil_id=sessao_atualizada.perfil_id,
            sessao_id=sessao_atualizada.session_id,
            resumo=sessao_atualizada.resumo,
            data_inicio=sessao_atualizada.data_inicio,
            mensagens_recentes=sessao_atualizada.mensagens[-self.janela:],
            agentes_chamados=sessao_atualizada.agentes_chamados,
        )

    def recuperar_historico(self, user_id: Optional[str], perfil_id: str):
        filtro = {
            'usuario_id' if user_id else 'perfil_id': (user_id or perfil_id),
            'resumo': {'$nin': ['', None]}
        }

        sessoes = self.repository.buscar_usuario(filtro, self.janela)
        return [
            MemoriaCtx(
                doc_id=sessao.id,
                usuario_id=sessao.usuario_id,
                perfil_id=sessao.perfil_id,
                sessao_id=sessao.session_id,
                resumo=sessao.resumo,
                data_inicio=sessao.data_inicio,
                mensagens_recentes=sessao.mensagens[-self.janela],
                agentes_chamados=sessao.agentes_chamados,
            )
            for sessao in sessoes
        ]