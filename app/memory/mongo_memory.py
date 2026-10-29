from typing import Dict, List, Optional

from app.memory.base import MemoriaCtx, MemoryStore
from app.llms.resumo_service import ResumoService
from app.repository.postgresql.perfil_repository import PerfilContext
from app.core.observabilidade import medir_span


class MongoMemory(MemoryStore):
    """
    Os `_id` da collection `sessao` são strings UUID (gerados em `Sessao.id`
    e gravados assim por `SessaoRepository.criar_sessao`), então nada aqui
    converte para `ObjectId`.

    Os métodos devolvem `None` quando não há sessão sobre a qual operar —
    quem chama decide o que fazer (ex.: 404 no endpoint de encerrar).
    """

    def __init__(self, repository, resumo_service: ResumoService, janela: int = 10):
        super().__init__(repository, resumo_service, janela)

    def _para_ctx(self, sessao) -> MemoriaCtx:
        return MemoriaCtx(
            doc_id=sessao.id,
            usuario_id=sessao.usuario_id,
            perfil_id=sessao.perfil_id,
            sessao_id=sessao.session_id,
            resumo=sessao.resumo,
            data_inicio=sessao.data_inicio,
            mensagens_recentes=sessao.mensagens[-self.janela:],
            agentes_chamados=sessao.agentes_chamados,
        )

    def preparar_turno(
        self,
        pergunta: str,
        perfil_ctx: PerfilContext,
        session_id: str,
        usuario_id: Optional[str] = None,
    ) -> Optional[MemoriaCtx]:
        """
        Garante que existe sessão ativa (criando se necessário) e registra a
        pergunta do usuário. Devolve o contexto que o resto do grafo pode
        precisar de forma atômica para evitar roundtrips desnecessários de rede.
        """
        with medir_span("mongo_preparar_turno", categoria="mongo"):
            sessao = self.repository.buscar_sessao_ativa(session_id)
            if sessao is None:
                identificador = getattr(perfil_ctx, 'identificador', None) or str(
                    perfil_ctx.cooperativa_id or perfil_ctx.empresa_id or perfil_ctx.perfil_id
                )
                doc_id = self.repository.criar_sessao(
                    session_id, identificador, usuario_id=usuario_id
                )
            else:
                doc_id = sessao.id

            if hasattr(self.repository, 'adicionar_mensagem_e_retornar'):
                sessao_atualizada = self.repository.adicionar_mensagem_e_retornar(
                    doc_id, role='usuario', content=pergunta
                )
            else:
                self.repository.adicionar_mensagem(doc_id, role='usuario', content=pergunta)
                sessao_atualizada = self.repository.buscar_por_id(doc_id)

            if sessao_atualizada is None:
                return None
            return self._para_ctx(sessao_atualizada)

    def registrar_resposta(
        self,
        resposta: str,
        session_id: str,
        agentes: Optional[List[str]] = None,
        meta: Optional[Dict] = None,
    ) -> Optional[MemoriaCtx]:
        """
        Registra a resposta final do turno de forma atômica.
        """
        with medir_span("mongo_registrar_resposta", categoria="mongo"):
            sessao = self.repository.buscar_sessao_ativa(session_id)
            if sessao is None:
                return None

            if hasattr(self.repository, 'adicionar_mensagem_e_retornar'):
                sessao_atualizada = self.repository.adicionar_mensagem_e_retornar(
                    sessao.id, role='assistente', content=resposta, agentes=agentes or [], meta=meta
                )
            else:
                self.repository.adicionar_mensagem(
                    sessao.id, role='assistente', content=resposta, agentes=agentes or [], meta=meta
                )
                sessao_atualizada = self.repository.buscar_por_id(sessao.id)

            if sessao_atualizada is None:
                return None
            return self._para_ctx(sessao_atualizada)

    def encerrar_sessao(self, session_id: str) -> Optional[MemoriaCtx]:
        """
        Decide se vale encerrar a sessão (só encerra se ela teve pelo menos
        uma mensagem), gera o resumo e persiste junto.
        """
        with medir_span("mongo_encerrar_sessao", categoria="mongo"):
            sessao = self.repository.buscar_sessao_ativa(session_id)
            if sessao is None or not sessao.mensagens:
                return None

            resumo = self.llm_service.gerar_resumo(sessao.mensagens, sessao.resumo)
            self.repository.marcar_encerrada(sessao.id, resumo=resumo)

            sessao_atualizada = self.repository.buscar_por_id(sessao.id)
            if sessao_atualizada is None:
                return None
            return self._para_ctx(sessao_atualizada)

    def recuperar_historico(self, user_id: Optional[str], perfil_id: str):
        with medir_span("mongo_recuperar_historico", categoria="mongo"):
            filtro = {
                'usuario_id' if user_id else 'perfil_id': (user_id or perfil_id),
                'resumo': {'$nin': ['', None]}
            }

            sessoes = self.repository.buscar_usuario(filtro, self.janela)
            return [self._para_ctx(sessao) for sessao in sessoes]
