from decimal import Decimal
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from psycopg_pool import ConnectionPool

from app.repository.base import Repository

log = logging.getLogger(__name__)


class AnaliticaRepository(Repository[Any, ConnectionPool]):
    """Repositorio Read-Only de Consultas Analiticas avancadas (Window Functions/DW) no PostgreSQL."""

    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def consultar_saude_financeira_esg(
        self,
        cooperativa_id: UUID,
        mes_referencia: Optional[str] = None,
        limite_meses: int = 6,
    ) -> List[Dict[str, Any]]:
        """Consulta indicadores socioeconomicos ESG (renda per capita, variacao MoM e media trimestral)."""
        query = '''
            SELECT
                mes_referencia,
                total_cooperados_beneficiados,
                montante_rateado_total,
                valor_medio_per_capita,
                menor_rateio,
                maior_rateio,
                valor_per_capita_mes_anterior,
                crescimento_mom_renda_pct,
                media_movel_trimestral_per_capita
            FROM vw_analitica_saude_financeira_rateio_esg
            WHERE cooperativa_id = %s
              AND (%s::text IS NULL OR TO_CHAR(mes_referencia, 'YYYY-MM') = %s)
            ORDER BY mes_referencia DESC
            LIMIT %s;
        '''
        params = [cooperativa_id, mes_referencia, mes_referencia, limite_meses]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'mes_referencia': str(row[0]),
                'total_cooperados_beneficiados': int(row[1]),
                'montante_rateado_total_reais': float(row[2] or Decimal('0')),
                'valor_medio_per_capita_reais': float(row[3] or Decimal('0')),
                'menor_rateio_reais': float(row[4] or Decimal('0')),
                'maior_rateio_reais': float(row[5] or Decimal('0')),
                'valor_per_capita_mes_anterior_reais': float(row[6]) if row[6] is not None else None,
                'crescimento_mom_renda_pct': float(row[7]) if row[7] is not None else None,
                'media_movel_trimestral_per_capita_reais': float(row[8] or Decimal('0')),
            }
            for row in linhas
        ]

    def consultar_desempenho_equipes_triagem(
        self,
        cooperativa_id: UUID,
        mes_referencia: Optional[str] = None,
        equipe_nome: Optional[str] = None,
        limite: int = 10,
    ) -> List[Dict[str, Any]]:
        """Consulta ranking de produtividade (DENSE_RANK) e deltas de rejeito de equipes de triagem."""
        query = '''
            SELECT
                equipe_id,
                nome_equipe,
                mes_referencia,
                total_lotes_triados,
                total_liquido_kg,
                total_rejeito_kg,
                taxa_rejeito_equipe_pct,
                ranking_produtividade,
                media_cooperativa_kg,
                desvio_vs_media_coop_pct,
                delta_taxa_rejeito_pct
            FROM vw_analitica_desempenho_equipes_triagem
            WHERE cooperativa_id = %s
              AND (%s::text IS NULL OR TO_CHAR(mes_referencia, 'YYYY-MM') = %s)
              AND (%s::text IS NULL OR LOWER(nome_equipe) LIKE LOWER('%%' || %s || '%%'))
            ORDER BY ranking_produtividade ASC
            LIMIT %s;
        '''
        params = [cooperativa_id, mes_referencia, mes_referencia, equipe_nome, equipe_nome, limite]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'equipe_id': str(row[0]),
                'nome_equipe': row[1],
                'mes_referencia': str(row[2]),
                'total_lotes_triados': int(row[3]),
                'total_liquido_kg': float(row[4] or Decimal('0')),
                'total_rejeito_kg': float(row[5] or Decimal('0')),
                'taxa_rejeito_equipe_pct': float(row[6] or Decimal('0')),
                'ranking_produtividade': int(row[7]),
                'media_cooperativa_kg': float(row[8] or Decimal('0')),
                'desvio_vs_media_coop_pct': float(row[9] or Decimal('0')),
                'delta_taxa_rejeito_pct': float(row[10] or Decimal('0')),
            }
            for row in linhas
        ]

    def consultar_anomalias_fluxo_coleta(
        self,
        cooperativa_id: UUID,
        cooperado_nome: Optional[str] = None,
        limite: int = 15,
    ) -> List[Dict[str, Any]]:
        """Consulta fluxo de coletas com variacoes atipicas de peso (LAG) e tempo de ciclo (LEAD)."""
        query = '''
            SELECT
                coleta_id,
                data_evento,
                nome_cooperado,
                tipo_coleta,
                quantidade_kg,
                quantidade_coleta_anterior_kg,
                variacao_vs_anterior_kg,
                intervalo_proxima_coleta_horas,
                acumulado_mes_cooperado_kg
            FROM vw_analitica_fluxo_coleta_anomalias
            WHERE cooperativa_id = %s
              AND (%s::text IS NULL OR LOWER(nome_cooperado) LIKE LOWER('%%' || %s || '%%'))
            ORDER BY data_evento DESC
            LIMIT %s;
        '''
        params = [cooperativa_id, cooperado_nome, cooperado_nome, limite]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'coleta_id': str(row[0]),
                'data_evento': row[1].isoformat() if row[1] else None,
                'nome_cooperado': row[2],
                'tipo_coleta': row[3],
                'quantidade_kg': float(row[4] or Decimal('0')),
                'quantidade_anterior_kg': float(row[5]) if row[5] is not None else None,
                'variacao_vs_anterior_kg': float(row[6]) if row[6] is not None else None,
                'intervalo_proxima_coleta_horas': float(row[7]) if row[7] is not None else None,
                'acumulado_mes_cooperado_kg': float(row[8] or Decimal('0')),
            }
            for row in linhas
        ]

    def consultar_cobertura_giro_estoque(
        self,
        cooperativa_id: UUID,
        material_nome: Optional[str] = None,
        status_capacidade: Optional[str] = None,
        limite: int = 10,
    ) -> List[Dict[str, Any]]:
        """Consulta analise de cobertura de estoque, ranking de volume (DENSE_RANK) e risco de capacidade."""
        query = '''
            SELECT
                material_id,
                nome_material,
                saldo_atual_kg,
                total_saidas_historico_kg,
                total_despachos,
                ranking_volume_estocado,
                share_estoque_cooperativa_pct,
                status_capacidade_armazenamento
            FROM vw_analitica_estoque_cobertura_giro
            WHERE cooperativa_id = %s
              AND (%s::text IS NULL OR LOWER(nome_material) LIKE LOWER('%%' || %s || '%%'))
              AND (%s::text IS NULL OR LOWER(status_capacidade_armazenamento) LIKE LOWER('%%' || %s || '%%'))
            ORDER BY ranking_volume_estocado ASC
            LIMIT %s;
        '''
        params = [cooperativa_id, material_nome, material_nome, status_capacidade, status_capacidade, limite]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'material_id': str(row[0]),
                'nome_material': row[1],
                'saldo_atual_kg': float(row[2] or Decimal('0')),
                'total_saidas_historico_kg': float(row[3] or Decimal('0')),
                'total_despachos': int(row[4] or 0),
                'ranking_volume_estocado': int(row[5]),
                'share_estoque_cooperativa_pct': float(row[6] or Decimal('0')),
                'status_capacidade_armazenamento': row[7],
            }
            for row in linhas
        ]

    def consultar_ranking_mercado_materiais(
        self,
        cooperativa_id: UUID,
        categoria_material: Optional[str] = None,
        mes_referencia: Optional[str] = None,
        limite: int = 10,
    ) -> List[Dict[str, Any]]:
        """Consulta market share relativo da cooperativa na triagem de residuos e variacao MoM."""
        query = '''
            SELECT
                mes_referencia,
                categoria_material,
                total_triado_kg,
                total_rejeito_kg,
                ranking_material_mes,
                share_material_mes_pct,
                volume_mes_anterior_kg,
                variacao_mom_pct
            FROM vw_fato_ranking_cooperativas_materiais
            WHERE cooperativa_id = %s
              AND (%s::text IS NULL OR LOWER(categoria_material) LIKE LOWER('%%' || %s || '%%'))
              AND (%s::text IS NULL OR TO_CHAR(mes_referencia, 'YYYY-MM') = %s)
            ORDER BY mes_referencia DESC, ranking_material_mes ASC
            LIMIT %s;
        '''
        params = [cooperativa_id, categoria_material, categoria_material, mes_referencia, mes_referencia, limite]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'mes_referencia': str(row[0]),
                'categoria_material': row[1],
                'total_triado_kg': float(row[2] or Decimal('0')),
                'total_rejeito_kg': float(row[3] or Decimal('0')),
                'ranking_material_mes': int(row[4]),
                'share_material_mes_pct': float(row[5] or Decimal('0')),
                'volume_mes_anterior_kg': float(row[6]) if row[6] is not None else None,
                'variacao_mom_pct': float(row[7]) if row[7] is not None else None,
            }
            for row in linhas
        ]

    def consultar_segmentacao_empresas_rfv(
        self,
        empresa_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """Consulta historico e segmentacao RFV (Recencia, Frequencia e Valor) da empresa compradora."""
        query = '''
            SELECT
                empresa_id,
                nome_empresa,
                total_pedidos_concluidos,
                volume_total_comprado_kg,
                faturamento_total_gerado,
                data_ultimo_pedido,
                recencia_dias,
                ranking_faturamento,
                quintil_faturamento,
                faixa_cliente,
                share_global_faturamento_pct
            FROM vw_analitica_segmentacao_empresas_rfv
            WHERE empresa_id = %s;
        '''
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [empresa_id])
                row = cur.fetchone()

        if not row:
            return None

        return {
            'empresa_id': str(row[0]),
            'nome_empresa': row[1],
            'total_pedidos_concluidos': int(row[2]),
            'volume_total_comprado_kg': float(row[3] or Decimal('0')),
            'faturamento_total_reais': float(row[4] or Decimal('0')),
            'data_ultimo_pedido': row[5].isoformat() if row[5] else None,
            'recencia_dias': int(row[6]),
            'ranking_faturamento': int(row[7]),
            'quintil_faturamento': int(row[8]),
            'faixa_cliente': row[9],
            'share_global_faturamento_pct': float(row[10] or Decimal('0')),
        }

    def consultar_avaliacoes_cooperativa(
        self,
        cooperativa_id: Optional[UUID] = None,
        cooperativa_nome: Optional[str] = None,
        nota_minima: int = 1,
        limite: int = 5,
    ) -> List[Dict[str, Any]]:
        """Consulta notas e comentarios qualitativos de compradores sobre a cooperativa."""
        query = '''
            SELECT
                a.avaliacao_id,
                c.cooperativa_id,
                c.nome AS nome_cooperativa,
                e.nome AS nome_empresa_avaliadora,
                a.nota,
                a.comentario,
                a.data_avaliacao
            FROM avaliacoes a
            JOIN perfis p_avaliado ON p_avaliado.perfil_id = a.avaliado_id
            JOIN cooperativas c ON c.cooperativa_id = p_avaliado.cooperativa_id
            JOIN perfis p_avaliador ON p_avaliador.perfil_id = a.avaliador_id
            JOIN empresas e ON e.empresa_id = p_avaliador.empresa_id
            WHERE a.nota >= %s
              AND (%s::uuid IS NULL OR c.cooperativa_id = %s)
              AND (%s::text IS NULL OR LOWER(c.nome) LIKE LOWER('%%' || %s || '%%'))
            ORDER BY a.data_avaliacao DESC
            LIMIT %s;
        '''
        params = [nota_minima, cooperativa_id, cooperativa_id, cooperativa_nome, cooperativa_nome, limite]

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                linhas = cur.fetchall()

        return [
            {
                'avaliacao_id': str(row[0]),
                'cooperativa_id': str(row[1]),
                'nome_cooperativa': row[2],
                'nome_empresa_avaliadora': row[3],
                'nota': int(row[4]),
                'comentario': row[5],
                'data_avaliacao': row[6].isoformat() if row[6] else None,
            }
            for row in linhas
        ]
