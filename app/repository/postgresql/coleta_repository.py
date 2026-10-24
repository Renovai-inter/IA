from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID

from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.repository.base import Snapshot, SnapshotRepository

import logging
log = logging.getLogger(__name__)


class Coleta(BaseModel):
    coleta_id: UUID
    cooperativa_id: UUID
    codigo: Optional[str] = None
    data_coleta: datetime
    status: str
    tipo_coleta: Optional[str] = 'EXTERNA'
    peso_estimado_kg: Optional[Decimal] = Decimal('0')
    peso_real_kg: Optional[Decimal] = Decimal('0')
    ponto_coleta: Optional[str] = None
    equipe_responsavel: Optional[str] = None
    observacoes: Optional[str] = None


class Triagem(BaseModel):
    triagem_id: UUID
    cooperativa_id: UUID
    coleta_id: Optional[UUID] = None
    data_triagem: datetime
    status: str
    peso_total_kg: Decimal
    peso_rejeito_kg: Optional[Decimal] = Decimal('0')
    equipe: Optional[str] = None
    observacoes: Optional[str] = None


class ColetaSnapshot(Snapshot[Coleta]):
    triagens: List[Triagem] = []


QUERY_COLETAS_POR_COOPERATIVA = """
    SELECT
        c.coleta_id,
        c.cooperativa_id,
        c.codigo,
        c.data_coleta,
        c.status,
        c.tipo_coleta,
        c.peso_estimado_kg,
        c.peso_real_kg,
        c.ponto_coleta,
        c.equipe_responsavel,
        c.observacoes
    FROM coletas c
    WHERE c.cooperativa_id = %s
    ORDER BY c.data_coleta DESC;
"""

QUERY_TRIAGENS_POR_COOPERATIVA = """
    SELECT
        t.triagem_id,
        t.cooperativa_id,
        t.coleta_id,
        t.data_triagem,
        t.status,
        t.peso_total_kg,
        t.peso_rejeito_kg,
        t.equipe,
        t.observacoes
    FROM triagens t
    WHERE t.cooperativa_id = %s
    ORDER BY t.data_triagem DESC;
"""


class ColetaRepository(SnapshotRepository[Coleta, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> ColetaSnapshot:
        log.debug('chegou no coleta_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_COLETAS_POR_COOPERATIVA, [cooperativa_id])
                itens_coleta = [Coleta(**item) for item in cur.fetchall()]

                cur.execute(QUERY_TRIAGENS_POR_COOPERATIVA, [cooperativa_id])
                itens_triagem = [Triagem(**item) for item in cur.fetchall()]

        return ColetaSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens_coleta,
            triagens=itens_triagem,
        )
