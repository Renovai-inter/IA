"""
Modelagem
collection: sessao
{
  '_id':               ObjectId(),
  'session_id':        'uuid-do-thread',    # == thread_id passado ao LangGraph (ChatRequest.session_id)
  'perfil_id':         'uuid-do-perfil',    # FK lógica pra perfis.id no Postgres
  'usuario_id':        'uuid' | None,
  'status':            'ativa' | 'encerrada',
  'data_inicio':       datetime,
  'data_atualizacao':  datetime,
  'data_encerramento': datetime | None,
  'resumo':            str | None,
  'agentes_chamados':  [str],
  'mensagens': [
    {
      'role':      'usuario' | 'assistente',
      'agentes':   [str],
      'content':   str,
      'timestamp': datetime,
      'meta':      dict | None,
    }
  ]
}
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field
from pymongo.database import Database

from app.repository.base import Repository


class Sessao(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: Optional[str] = Field(default=None, alias='_id')
    session_id: str
    perfil_id: str
    usuario_id: Optional[str] = None
    status: Literal['ativa', 'encerrada']
    data_inicio: datetime
    data_atualizacao: datetime
    data_encerramento: Optional[datetime] = None
    resumo: Optional[str] = None
    agentes_chamados: List[str] = []
    mensagens: List[Dict] = []


class SessaoRepository(Repository[Sessao]):
    """
    Camada mecânica: só sabe conversar com a collection `sessao`. Não decide
    QUANDO criar, encerrar ou resumir uma sessão — quem decide isso é
    `memory/mongo_memory.py`. Cada método aqui recebe a decisão já tomada
    por quem chama e só executa o comando correspondente no Mongo.

    Observação: `db` precisa ser um `pymongo.database.Database`, não um
    `MongoClient` — `self._db['sessao']` só retorna a Collection certa se
    `self._db` já for o Database (indexar um MongoClient por nome de string
    devolve um Database, não uma Collection, e ia quebrar no primeiro
    `create_index`).
    """

    def __init__(self, db: Database):
        super().__init__(db)
        self._col_sessao = None
        self._indexes_created = False

    def _get_collection(self):
        if self._col_sessao is not None:
            return self._col_sessao

        self._col_sessao = self._db['sessao']

        if not self._indexes_created:
            self._col_sessao.create_index('session_id')
            self._col_sessao.create_index('data_inicio')
            self._indexes_created = True

        return self._col_sessao

    def _agora(self) -> datetime:
        return datetime.now(timezone.utc)

    def buscar_sessao_ativa(self, session_id: str) -> Optional[Sessao]:
        """Lê a sessão com status 'ativa' mais recente para este session_id, se existir."""
        doc = self._get_collection().find_one(
            {'session_id': session_id, 'status': 'ativa'},
            sort=[('data_inicio', -1)],
        )
        if not doc:
            return None

        doc['_id'] = str(doc['_id'])
        return Sessao(**doc)

    def buscar_por_id(self, doc_id: ObjectId) -> Optional[Sessao]:
        doc = self._get_collection().find_one({'_id': doc_id})
        if not doc:
            return None

        doc['_id'] = str(doc['_id'])
        return Sessao(**doc)

    def criar_sessao(self, session_id: str, perfil_id: str, usuario_id: Optional[str] = None) -> ObjectId:
        """Insere um novo documento"""
        agora = self._agora()
        novo_doc = {
            'session_id': session_id,
            'perfil_id': perfil_id,
            'usuario_id': usuario_id,
            'status': 'ativa',
            'data_inicio': agora,
            'data_atualizacao': agora,
            'data_encerramento': None,
            'resumo': None,
            'agentes_chamados': [],
            'mensagens': [],
        }
        resultado = self._get_collection().insert_one(novo_doc)
        return resultado.inserted_id

    def adicionar_mensagem(
        self,
        doc_id: ObjectId,
        role: str,
        content: str,
        agentes: Optional[List[str]] = None,
        meta: Optional[Dict] = None,
    ) -> None:
        """Dado o _id de uma sessão já existente, só faz o push da mensagem."""
        mensagem = {
            'role': role,
            'content': content,
            'timestamp': self._agora(),
        }
        if agentes is not None:
            mensagem['agentes'] = agentes
        if meta is not None:
            mensagem['meta'] = meta

        self._get_collection().update_one(
            {'_id': doc_id},
            {
                '$set': {'data_atualizacao': self._agora()},
                '$push': {'mensagens': mensagem},
                '$addToSet': {'agentes_chamados': {'$each': agentes if agentes else []}},
            },
        )

    def marcar_encerrada(self, doc_id: ObjectId, resumo: Optional[str] = None) -> None:
        """Dado o _id de uma sessão, marca como encerrada. Não decide SE deveria encerrar."""
        agora = self._agora()
        update = {'status': 'encerrada', 'data_atualizacao': agora, 'data_encerramento': agora}
        if resumo is not None:
            update['resumo'] = resumo

        self._get_collection().update_one({'_id': doc_id}, {'$set': update})

    def buscar_usuario(self, filtro, limite = 3) -> List[Sessao]:
        docs = (
            self._get_collection()
                .find(filtro)
                .limit(limite)
        )

        docs = [{**doc, '_id': str(doc['_id'])} for doc in docs]
        return [Sessao(**doc) for doc in docs]