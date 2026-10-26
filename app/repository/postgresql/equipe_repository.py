from datetime import datetime
from typing import List, Optional
from uuid import UUID
import logging

from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from app.repository.base import Snapshot, SnapshotRepository

log = logging.getLogger(__name__)


class Equipe(BaseModel):
    equipe_id: UUID
    cooperativa_id: UUID
    gestor_id: UUID
    nome_gestor: Optional[str] = None
    nome: Optional[str] = None
    descricao: Optional[str] = None
    data_criacao: datetime
    esta_ativa: bool
    cooperado_ids: List[UUID] = Field(default_factory=list)


class EquipeSnapshot(Snapshot[Equipe]):
    pass


QUERY_EQUIPES_POR_COOPERATIVA = """
    SELECT
        e.equipe_id,
        g.cooperativa_id,
        e.gestor_id,
        ug.nome AS nome_gestor,
        e.nome,
        e.descricao,
        e.data_criacao,
        e.esta_ativa,
        COALESCE(
            ARRAY_AGG(ec.cooperado_id) FILTER (WHERE ec.cooperado_id IS NOT NULL),
            ARRAY[]::uuid[]
        ) AS cooperado_ids
    FROM equipes e
    INNER JOIN funcionarios g
        ON e.gestor_id = g.funcionario_id
    LEFT JOIN usuarios ug
        ON g.usuario_id = ug.usuario_id
    LEFT JOIN equipes_cooperados ec
        ON e.equipe_id = ec.equipe_id
    WHERE g.cooperativa_id = %s
    GROUP BY
        e.equipe_id,
        g.cooperativa_id,
        e.gestor_id,
        ug.nome,
        e.nome,
        e.descricao,
        e.data_criacao,
        e.esta_ativa
    ORDER BY e.nome;
"""


class EquipeRepository(SnapshotRepository[Equipe, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> EquipeSnapshot:
        log.debug('chegou no equipe_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_EQUIPES_POR_COOPERATIVA, [cooperativa_id])
                itens = [Equipe(**item) for item in cur.fetchall()]

        return EquipeSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )
