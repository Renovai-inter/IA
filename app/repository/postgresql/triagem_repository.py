from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID
import logging

from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.repository.base import Snapshot, SnapshotRepository

log = logging.getLogger(__name__)


class Triagem(BaseModel):
    triagem_id: UUID
    cooperativa_id: UUID
    equipe_id: UUID
    nome_equipe: Optional[str] = None
    coleta_id: UUID
    material_id: UUID
    nome_categoria: Optional[str] = None
    categoria_pai_id: Optional[UUID] = None
    nome_categoria_pai: Optional[str] = None
    data_triagem: datetime
    status_id: Optional[UUID] = None
    status_atual: Optional[str] = None
    quantidade_kg: Decimal
    quantidade_rejeito_kg: Optional[Decimal] = Decimal('0')
    imagem_url: Optional[str] = None


class TriagemSnapshot(Snapshot[Triagem]):
    pass


QUERY_TRIAGENS_POR_COOPERATIVA = """
    SELECT
        t.evento_id AS triagem_id,
        g.cooperativa_id,
        t.equipe_id,
        e.nome AS nome_equipe,
        t.coleta_id,
        t.material_id,
        cm.nome_categoria,
        cm.categoria_pai_id,
        cm2.nome_categoria AS nome_categoria_pai,
        t.data_evento AS data_triagem,
        t.status_id,
        s.status_atual,
        t.quantidade_kg,
        t.quantidade_rejeito_kg,
        t.imagem_url
    FROM triagens t
    INNER JOIN equipes e
        ON t.equipe_id = e.equipe_id
    INNER JOIN funcionarios g
        ON e.gestor_id = g.funcionario_id
    INNER JOIN materiais m
        ON t.material_id = m.material_id
    INNER JOIN categorias_materiais cm
        ON m.categoria_id = cm.categoria_id
    LEFT JOIN categorias_materiais cm2
        ON cm.categoria_pai_id = cm2.categoria_id
    LEFT JOIN status s
        ON t.status_id = s.status_id
    WHERE g.cooperativa_id = %s
    ORDER BY t.data_evento DESC;
"""


class TriagemRepository(SnapshotRepository[Triagem, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> TriagemSnapshot:
        log.debug('chegou no triagem_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_TRIAGENS_POR_COOPERATIVA, [cooperativa_id])
                itens = [Triagem(**item) for item in cur.fetchall()]

        return TriagemSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )
