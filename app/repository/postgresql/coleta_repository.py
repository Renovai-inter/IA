from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID
import logging

from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.repository.base import Snapshot, SnapshotRepository

log = logging.getLogger(__name__)


class Coleta(BaseModel):
    coleta_id: UUID
    cooperativa_id: UUID
    cooperado_id: UUID
    cooperado_responsavel: Optional[str] = None
    rota_id: Optional[UUID] = None
    data_coleta: datetime
    status_id: Optional[UUID] = None
    status_atual: Optional[str] = None
    tipo_coleta: Optional[str] = None
    quantidade_kg: Decimal
    imagem_url: Optional[str] = None


class ColetaSnapshot(Snapshot[Coleta]):
    pass


QUERY_COLETAS_POR_COOPERATIVA = """
    SELECT
        c.evento_id AS coleta_id,
        f.cooperativa_id,
        c.cooperado_id,
        u.nome AS cooperado_responsavel,
        c.rota_id,
        c.data_evento AS data_coleta,
        c.status_id,
        s.status_atual,
        c.tipo_coleta,
        c.quantidade_kg,
        c.imagem_url
    FROM coletas c
    INNER JOIN funcionarios f
        ON c.cooperado_id = f.funcionario_id
    LEFT JOIN status s
        ON c.status_id = s.status_id
    LEFT JOIN usuarios u
        ON f.usuario_id = u.usuario_id
    WHERE f.cooperativa_id = %s
    ORDER BY c.data_evento DESC;
"""


class ColetaRepository(SnapshotRepository[Coleta, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> ColetaSnapshot:
        log.debug('chegou no coleta_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_COLETAS_POR_COOPERATIVA, [cooperativa_id])
                itens = [Coleta(**item) for item in cur.fetchall()]

        return ColetaSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )
