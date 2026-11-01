from decimal import Decimal
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.repository.base import Repository

log = logging.getLogger(__name__)


class VendaItem(BaseModel):
    material_id: UUID
    nome_material: str
    volume_total_kg: Decimal
    faturamento_total: Decimal
    total_pedidos: int


class CurvaAbcItem(BaseModel):
    material_id: UUID
    nome_material: str
    volume_total_kg: Decimal
    faturamento_total: Decimal
    share_receita_pct: Decimal
    share_acumulado_pct: Decimal
    classificacao_abc: str


class VendaRepository(Repository[Any, ConnectionPool]):
    """Repositorio Read-Only de Vendas, Faturamento e Curva ABC no PostgreSQL."""

    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def consultar_faturamento_vendas(
        self,
        cooperativa_id: UUID,
        mes_referencia: Optional[str] = None,
        material_nome: Optional[str] = None,
        limite: int = 10,
    ) -> List[Dict[str, Any]]:
        """Consulta agregada em runtime de vendas faturadas para a cooperativa especificada."""
        query = '''
            SELECT
                v.material_id,
                m.nome_categoria AS nome_material,
                SUM(v.quantidade_kg) AS volume_total_kg,
                SUM(v.valor_total) AS faturamento_total,
                COUNT(DISTINCT v.pedido_id) AS total_pedidos
            FROM vw_fato_vendas v
            JOIN vw_dim_material m ON m.material_id = v.material_id
            WHERE v.cooperativa_id = %s
              AND (%s::text IS NULL OR TO_CHAR(v.data_pedido, 'YYYY-MM') = %s)
              AND (%s::text IS NULL OR LOWER(m.nome_categoria) LIKE LOWER('%%' || %s || '%%'))
            GROUP BY v.material_id, m.nome_categoria
            ORDER BY faturamento_total DESC
            LIMIT %s;
        '''
        params = [cooperativa_id, mes_referencia, mes_referencia, material_nome, material_nome, limite]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'material_id': str(row[0]),
                'nome_material': row[1],
                'volume_total_kg': float(row[2] or Decimal('0')),
                'faturamento_total_reais': float(row[3] or Decimal('0')),
                'total_pedidos': int(row[4] or 0),
            }
            for row in linhas
        ]

    def consultar_curva_abc(
        self,
        cooperativa_id: UUID,
        classificacao: Optional[str] = None,
        limite: int = 10,
    ) -> List[Dict[str, Any]]:
        """Consulta em runtime da curva ABC de faturamento de materiais da cooperativa."""
        query = '''
            SELECT
                material_id,
                nome_material,
                volume_total_kg,
                faturamento_total,
                share_receita_pct,
                share_acumulado_pct,
                classificacao_abc
            FROM vw_analitica_curva_abc_materiais
            WHERE cooperativa_id = %s
              AND (%s::text IS NULL OR LOWER(classificacao_abc) LIKE LOWER('%%' || %s || '%%'))
            ORDER BY faturamento_total DESC
            LIMIT %s;
        '''
        params = [cooperativa_id, classificacao, classificacao, limite]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'material_id': str(row[0]),
                'nome_material': row[1],
                'volume_total_kg': float(row[2] or Decimal('0')),
                'faturamento_total_reais': float(row[3] or Decimal('0')),
                'share_receita_pct': float(row[4] or Decimal('0')),
                'share_acumulado_pct': float(row[5] or Decimal('0')),
                'classificacao_abc': row[6],
            }
            for row in linhas
        ]
