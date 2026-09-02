from app.repository.base import Repository, Snapshot

from pydantic import BaseModel

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

class Material(BaseModel):
    material_id: UUID
    categoria_id: UUID
    nome_categoria: str
    cooperativa_id: Optional[UUID]
    nome_cooperativa: Optional[str]
    preco_sugerido: Optional[Decimal]
    esta_disponivel: bool
    imagem_url: Optional[str] = None

class MaterialSnapshot[Material](Snapshot):
    cooperativa_id: UUID
    capturado_em: datetime
    itens: list[Material]

class MaterialRepository[Material](Repository):
    def __init__(self, db):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> Snapshot:
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                query = """
                SELECT 
                    m.material_id,
                    m.categoria_id,
                    cm.nome_categoria,
                    m.cooperativa_id,
                    co.nome AS nome_cooperativa,
                    m.preco_sugerido,
                    m.esta_disponivel,
                    m.imagem_url
                FROM materiais m
                INNER JOIN categorias_materiais cm 
                    ON m.categoria_id = cm.categoria_id
                LEFT JOIN cooperativas co 
                    ON m.cooperativa_id = co.cooperativa_id
                WHERE cooperativa_id = %s;
                """
                params = [cooperativa_id]
                
                cur.execute(query, params)
                
                colnames = [desc[0] for desc in cur.description]
                itens = []
                for row in cur.fetchall():
                    row_dict = dict(zip(colnames, row))
                    itens.append(Material(**row_dict))
                
                return MaterialSnapshot(
                    cooperativa_id=cooperativa_id,
                    capturado_em=datetime.now(),
                    itens=itens
                )