from app.repository.base import Repository
from app.repository.postgresql.db import get_conn

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

class MaterialSnapshot(BaseModel):
    cooperativa_id: UUID
    capturado_em: datetime
    itens: list[Material]

class MaterialRepository[Material](Repository):
    def __init__(self, perfil_id, entity):
        super().__init__(perfil_id, Material)

    def _get_cooperativa_id(self) -> UUID:
        conn = get_conn()
        cur = conn.cursor()

        query = "SELECT cooperativa_id FROM perfis"
        params = [self.perfil_id]

        cur.execute(query, params)
        return UUID(cur.fetchone()[0])

    def get_snapshot(self) -> Material:
        conn = get_conn()
        cur = conn.cursor()

        cooperativa_id = self._get_cooperativa_id()

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