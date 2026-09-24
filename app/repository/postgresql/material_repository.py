from psycopg_pool import ConnectionPool

from app.repository.base import Snapshot, SnapshotRepository

from pydantic import BaseModel

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

class Material(BaseModel):
    material_id: UUID
    categoria_id: UUID
    nome_categoria: str
    categoria_pai_id: Optional[UUID]
    nome_categoria_pai: Optional[str]
    cooperativa_id: Optional[UUID]
    nome_cooperativa: Optional[str]
    preco_sugerido: Optional[Decimal]
    esta_disponivel: bool
    imagem_url: Optional[str] = None

class MaterialSnapshot(Snapshot[Material]):
    pass

_QUERY_MATERIAL_POR_COOPERATIVA = """
    SELECT 
        m.material_id,
        m.categoria_id,
        cm.nome_categoria,
        cm.categoria_pai_id,
        cm2.nome_categoria AS nome_categoria_pai,
        m.cooperativa_id,
        co.nome AS nome_cooperativa,
        m.preco_sugerido,
        m.esta_disponivel,
        m.imagem_url
    FROM materiais m
    INNER JOIN categorias_materiais cm 
        ON m.categoria_id = cm.categoria_id
    LEFT JOIN categorias_materiais cm2
        ON cm.categoria_pai_id = cm2.categoria_id
    LEFT JOIN cooperativas co 
        ON m.cooperativa_id = co.cooperativa_id
    WHERE co.cooperativa_id = %s;
"""

class MaterialRepository(SnapshotRepository[Material, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> MaterialSnapshot:
        print('[DEBUG]: chegou no material_repository e tirou snapshot')
        with self._db.connection() as conn:
            with conn.cursor() as cur:                
                cur.execute(_QUERY_MATERIAL_POR_COOPERATIVA, [cooperativa_id])
                itens = [Material(**material) for material in cur.fetchall()]
                
                return MaterialSnapshot(
                    cooperativa_id=cooperativa_id,
                    capturado_em=datetime.now(),
                    itens=itens
                )