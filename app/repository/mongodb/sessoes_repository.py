"""
Modelagem
collection: sessoes
{
  '_id':               ObjectId(),
  'session_id':        'uuid-do-thread',    # == thread_id passado ao LangGraph (ChatRequest.session_id)
  'perfil_id':         'uuid-do-perfil',    # FK lógica pra perfis.id no Postgres
  'usuario_id':        'uuid' | None,       # ver observação sobre esse campo na mensagem de review
  'status':            'ativa' | 'encerrada',
  'data_inicio':       datetime,
  'data_atualizacao':  datetime,
  'data_encerramento': datetime | None,
  'resumo':            str | None,
  'agentes_chamados':  [str],               # espelha o reducer agentes_chamados do GraphState
  'mensagens': [
    {
      'role':      'usuario' | 'assistente',
      'agente':    str | None,              # qual especialista gerou; None pro turno do usuário
      'content':   str,
      'timestamp': datetime,
      'meta':      dict | None,             # custo/latência por chamada de LLM (Fase 5 - observabilidade)
    }
  ]
}
"""

from app.repository.base import Repository

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field
from pymongo.database import Database


class Sessoes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: Optional[str] = Field(default=None, alias="_id")
    session_id: str
    perfil_id: str
    usuario_id: Optional[str] = None
    status: Literal["ativa", "encerrada"]
    data_inicio: datetime
    data_atualizacao: datetime
    data_encerramento: Optional[datetime] = None
    resumo: Optional[str] = None
    agentes_chamados: List[str] = []
    mensagens: List[Dict] = []


class SessoesRepository(Repository[Sessoes]):
    def __init__(self, db: MongoClient):
        super().__init__(db)

        self._col_sessoes = None
        self._indexes_created = False
        self._sessoes_ativas: Dict[str, ObjectId] = {}

    def _get_collection(self):
        if self._col_sessoes is not None:
            return self._col_sessoes

        self._col_sessoes = self._db["sessoes"]

        if not self._indexes_created:
            self._col_sessoes.create_index("session_id")
            self._col_sessoes.create_index("data_inicio")
            self._indexes_created = True

        return self._col_sessoes

    def _agora(self) -> datetime:
        return datetime.now(timezone.utc)

    def _doc_id_da_sessao(self, session_id: str) -> Optional[ObjectId]:
        """
        Descobre o _id do documento da sessão EM ANDAMENTO (status == 'ativa') para este
        session_id. Usa o campo `status` (não a presença/ausência de `resumo`, que é
        preenchido pelo Orquestrador em outro momento e não indica se a sessão está aberta).
        """
        doc_id = self._sessoes_ativas.get(session_id)
        if doc_id:
            return doc_id

        col = self._get_collection()
        doc = col.find_one(
            {"session_id": session_id, "status": "ativa"},
            {"_id": 1},
            sort=[("data_inicio", -1)],
        )
        if not doc:
            return None

        doc_id = doc["_id"]
        self._sessoes_ativas[session_id] = doc_id
        return doc_id

    # ==============================================================================
    # FUNÇÕES PRINCIPAIS
    # ==============================================================================

    def iniciar_sessao(self, session_id: str, perfil_id: str, usuario_id: Optional[str] = None) -> ObjectId:
        """
        Garante que existe um documento de sessão aberta no MongoDB para este session_id
        e retorna o _id do documento (novo ou já existente).
        """
        doc_id_existente = self._doc_id_da_sessao(session_id)
        if doc_id_existente:
            return doc_id_existente

        agora = self._agora()
        novo_doc = {
            "session_id": session_id,
            "perfil_id": perfil_id,
            "usuario_id": usuario_id,
            "status": "ativa",
            "data_inicio": agora,
            "data_atualizacao": agora,
            "data_encerramento": None,
            "resumo": None,
            "agentes_chamados": [],
            "mensagens": [],
        }

        resultado = self._get_collection().insert_one(novo_doc)
        doc_id = resultado.inserted_id
        self._sessoes_ativas[session_id] = doc_id
        return doc_id


    def salvar_mensagem(
        self,
        session_id: str,
        perfil_id: str,
        role: str,
        content: str,
        usuario_id: Optional[str] = None,
        agente: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> None:
        """
        Adiciona uma mensagem ao array de mensagens da sessão ativa, criando a sessão
        se ainda não existir. `perfil_id` é obrigatório aqui porque `iniciar_sessao`
        precisa dele para criar o documento na primeira chamada.
        """
        doc_id = self.iniciar_sessao(session_id, perfil_id, usuario_id=usuario_id)

        mensagem = {
            "role": role,
            "agente": agente,
            "content": content,
            "timestamp": self._agora(),
        }
        if meta is not None:
            mensagem["meta"] = meta

        self._get_collection().update_one(
            {"_id": doc_id},
            {
                "$set": {"data_atualizacao": self._agora()},
                "$push": {"mensagens": mensagem},
            },
        )


    def registrar_agente_chamado(self, session_id: str, agente: str) -> None:
        """
        Acrescenta um agente à lista `agentes_chamados` da sessão ativa, sem duplicar
        (espelha o reducer `agentes_chamados` do GraphState). Chame isso a partir do
        Roteador/Orquestrador quando um especialista for acionado no turno.
        """
        doc_id = self._doc_id_da_sessao(session_id)
        if not doc_id:
            return

        self._get_collection().update_one(
            {"_id": doc_id},
            {"$addToSet": {"agentes_chamados": agente}},
        )


    def encerrar_sessao(self, session_id: str, resumo: Optional[str] = None) -> None:
        """
        Marca a sessão ativa como encerrada. Se `resumo` for passado (ex: gerado pelo
        Orquestrador), grava também nesse momento.
        """
        doc_id = self._doc_id_da_sessao(session_id)
        if not doc_id:
            return

        col = self._get_collection()
        doc = col.find_one({"_id": doc_id})

        if not doc or not doc.get("mensagens"):
            self._sessoes_ativas.pop(session_id, None)
            return

        agora = self._agora()
        update = {"status": "encerrada", "data_atualizacao": agora, "data_encerramento": agora}
        if resumo is not None:
            update["resumo"] = resumo

        col.update_one({"_id": doc_id}, {"$set": update})
        self._sessoes_ativas.pop(session_id, None)


    def obter_sessao(self, session_id: str) -> Optional[Sessoes]:
        """
        Leitura validada (via Pydantic) da sessão ativa mais recente — útil pro Guardrail
        de entrada consultar histórico/contexto antes de rotear.
        """
        doc_id = self._doc_id_da_sessao(session_id)
        if not doc_id:
            return None

        doc = self._get_collection().find_one({"_id": doc_id})
        if not doc:
            return None

        doc["_id"] = str(doc["_id"])
        return Sessoes(**doc)