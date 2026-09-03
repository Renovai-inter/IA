# repository/postgresql/estoque_repository.py
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.repository.base import Repository, Snapshot


class Estoque(BaseModel):
    estoque_id: UUID
    cooperativa_id: UUID
    nome_cooperativa: str
    material_id: UUID
    nome_categoria: str
    quantidade_kg: Decimal
    data_atualizacao: datetime


class EstoqueSnapshot(Snapshot[Estoque]):
    pass


QUERY_ESTOQUE_POR_COOPERATIVA = """
    SELECT
        e.estoque_id,
        e.cooperativa_id,
        co.nome AS nome_cooperativa,
        e.material_id,
        cm.nome_categoria,
        e.quantidade_kg,
        e.data_atualizacao
    FROM estoques e
    INNER JOIN materiais m
        ON e.material_id = m.material_id
    INNER JOIN categorias_materiais cm
        ON m.categoria_id = cm.categoria_id
    INNER JOIN cooperativas co
        ON e.cooperativa_id = co.cooperativa_id
    WHERE e.cooperativa_id = %s
    ORDER BY cm.nome_categoria;
"""


class EstoqueRepository(Repository[Estoque]):
    def get_snapshot(self, cooperativa_id: UUID) -> EstoqueSnapshot:
        print('[DEBUG]: chegou no material_repository e tirou snapshot')
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_ESTOQUE_POR_COOPERATIVA, [cooperativa_id])
                itens = [Estoque(**item) for item in cur.fetchall()]

        return EstoqueSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )