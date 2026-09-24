# repository/postgresql/movimentacao_estoque_repository.py
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from psycopg_pool import ConnectionPool
from pydantic import BaseModel
from typing import Optional

from app.repository.base import Snapshot, SnapshotRepository


class MovimentacaoEstoque(BaseModel):
    movimentacao_id: UUID
    estoque_id: UUID
    cooperativa_id: UUID
    material_id: UUID
    nome_categoria: str
    categoria_pai_id: Optional[UUID]
    nome_categoria_pai: Optional[str]
    triagem_id: Optional[UUID]
    item_id: Optional[UUID]
    quantidade_kg: Decimal
    tipo_movimentacao: str
    data_movimentacao: datetime


class MovimentacaoEstoqueSnapshot(Snapshot[MovimentacaoEstoque]):
    pass


QUERY_MOVIMENTACOES_POR_COOPERATIVA = """
    SELECT
        mv.movimentacao_id,
        mv.estoque_id,
        e.cooperativa_id,
        e.material_id,
        cm.nome_categoria,
        cm.categoria_pai_id,
        cm2.nome_categoria AS nome_categoria_pai,
        mv.triagem_id,
        mv.item_id,
        mv.quantidade_kg,
        mv.tipo_movimentacao,
        mv.data_movimentacao
    FROM movimentacoes_estoques mv
    INNER JOIN estoques e
        ON mv.estoque_id = e.estoque_id
    INNER JOIN materiais m
        ON e.material_id = m.material_id
    INNER JOIN categorias_materiais cm
        ON m.categoria_id = cm.categoria_id
    LEFT JOIN categorias_materiais cm2
        ON cm.categoria_pai_id = cm2.categoria_id
    WHERE e.cooperativa_id = %s
    ORDER BY mv.data_movimentacao DESC;
"""


class MovimentacaoEstoqueRepository(SnapshotRepository[MovimentacaoEstoque, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> MovimentacaoEstoqueSnapshot:
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_MOVIMENTACOES_POR_COOPERATIVA, [cooperativa_id])
                itens = [MovimentacaoEstoque(**item) for item in cur.fetchall()]

        return MovimentacaoEstoqueSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )