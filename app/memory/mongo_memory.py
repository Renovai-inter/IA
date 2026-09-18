from typing import Dict, Optional

from bson import ObjectId

from app.memory.base import MemoriaCtx, MemoryStore
from app.memory.resumo_service import ResumoService


class MongoMemory(MemoryStore):
    def __init__(self, repository, resumo_service: ResumoService, janela_mensagens: int = 10):
        super().__init__(repository, janela_mensagens)
        self.resumo_service = resumo_service

    def preparar_turno(
        self,
        session_id: str,
        perfil_id: str,
        pergunta: str,
        usuario_id: Optional[str] = None,
    ) -> MemoriaCtx:
        """
        Chamado pelo Guardrail de entrada antes do roteamento: garante que
        existe sessão ativa (criando se necessário) e registra a pergunta
        do usuário. Devolve o contexto que o resto do grafo pode precisar.
        """
        sessao = self.repository.buscar_sessao_ativa(session_id)
        if sessao is None:
            doc_id = self.repository.criar_sessao(session_id, perfil_id, usuario_id=usuario_id)
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
        session_id: str,
        resposta: str,
        agente: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> None:
        """Chamado pelo Orquestrador ao final do turno."""
        sessao = self.repository.buscar_sessao_ativa(session_id)
        if sessao is None:
            return

        doc_id = ObjectId(sessao.id)
        self.repository.adicionar_mensagem(doc_id, role="assistente", content=resposta, agente=agente, meta=meta)

    def encerrar_e_resumir(self, session_id: str, resumo: str) -> None:
        """
        Decide se vale encerrar a sessão (só encerra se ela teve pelo menos
        uma mensagem) e, quando um resumo for fornecido, persiste junto.
        """
        sessao = self.repository.buscar_sessao_ativa(session_id)
        if sessao is None or not sessao.mensagens:
            return ''

        resumo = ''
        resumo = self._resumo_service.gerar_resumo(sessao.mensagens, sessao.resumo)

        self.repository.encerrar_sessao(ObjectId(sessao.id), resumo=resumo)
        return resumo