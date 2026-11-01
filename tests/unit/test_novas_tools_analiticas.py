import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from app.tools.financeiro_tools import FinanceiroToolkit
from app.tools.coleta_tools import ColetaToolkit
from app.tools.material_estoque_tools import MaterialEstoqueToolkit
from app.tools.recomendacao_tools import RecomendacaoToolkit


@pytest.fixture
def fake_repositories():
    coop_id = uuid4()
    emp_id = uuid4()

    mock_venda_repo = MagicMock()
    mock_venda_repo.consultar_faturamento_vendas.return_value = [
        {
            'material_id': str(uuid4()),
            'nome_material': 'PET',
            'volume_total_kg': 1500.0,
            'faturamento_total_reais': 4500.0,
            'total_pedidos': 3,
        }
    ]
    mock_venda_repo.consultar_curva_abc.return_value = [
        {
            'material_id': str(uuid4()),
            'nome_material': 'PET',
            'volume_total_kg': 1500.0,
            'faturamento_total_reais': 4500.0,
            'share_receita_pct': 75.0,
            'share_acumulado_pct': 75.0,
            'classificacao_abc': 'Classe A (Alto Impacto / 80%)',
        }
    ]

    mock_analitica_repo = MagicMock()
    mock_analitica_repo.consultar_saude_financeira_esg.return_value = [
        {
            'mes_referencia': '2026-09-01',
            'total_cooperados_beneficiados': 12,
            'montante_rateado_total_reais': 25000.0,
            'valor_medio_per_capita_reais': 2083.33,
            'menor_rateio_reais': 1800.0,
            'maior_rateio_reais': 2400.0,
            'valor_per_capita_mes_anterior_reais': 1950.0,
            'crescimento_mom_renda_pct': 6.84,
            'media_movel_trimestral_per_capita_reais': 2010.0,
        }
    ]
    mock_analitica_repo.consultar_desempenho_equipes_triagem.return_value = [
        {
            'equipe_id': str(uuid4()),
            'nome_equipe': 'Equipe Alfa',
            'mes_referencia': '2026-09-01',
            'total_lotes_triados': 40,
            'total_liquido_kg': 8500.0,
            'total_rejeito_kg': 250.0,
            'taxa_rejeito_equipe_pct': 2.86,
            'ranking_produtividade': 1,
            'media_cooperativa_kg': 7200.0,
            'desvio_vs_media_coop_pct': 18.06,
            'delta_taxa_rejeito_pct': -1.5,
        }
    ]
    mock_analitica_repo.consultar_anomalias_fluxo_coleta.return_value = [
        {
            'coleta_id': str(uuid4()),
            'data_evento': '2026-09-10T14:30:00',
            'nome_cooperado': 'Carlos Silva',
            'tipo_coleta': 'EXTERNA',
            'quantidade_kg': 650.0,
            'quantidade_anterior_kg': 320.0,
            'variacao_vs_anterior_kg': 330.0,
            'intervalo_proxima_coleta_horas': 4.5,
            'acumulado_mes_cooperado_kg': 1850.0,
        }
    ]
    mock_analitica_repo.consultar_cobertura_giro_estoque.return_value = [
        {
            'material_id': str(uuid4()),
            'nome_material': 'Papelão',
            'saldo_atual_kg': 4200.0,
            'total_saidas_historico_kg': 12000.0,
            'total_despachos': 8,
            'ranking_volume_estocado': 1,
            'share_estoque_cooperativa_pct': 48.5,
            'status_capacidade_armazenamento': 'Alto Volume / Atenção a Espaço Físico',
        }
    ]
    mock_analitica_repo.consultar_ranking_mercado_materiais.return_value = [
        {
            'mes_referencia': '2026-09-01',
            'categoria_material': 'Latinha',
            'total_triado_kg': 2100.0,
            'total_rejeito_kg': 50.0,
            'ranking_material_mes': 1,
            'share_material_mes_pct': 34.5,
            'volume_mes_anterior_kg': 1800.0,
            'variacao_mom_pct': 16.67,
        }
    ]
    mock_analitica_repo.consultar_segmentacao_empresas_rfv.return_value = {
        'empresa_id': str(emp_id),
        'nome_empresa': 'Indústria Circular SA',
        'total_pedidos_concluidos': 15,
        'volume_total_comprado_kg': 48000.0,
        'faturamento_total_reais': 160000.0,
        'data_ultimo_pedido': '2026-09-20T10:00:00',
        'recencia_dias': 10,
        'ranking_faturamento': 1,
        'quintil_faturamento': 1,
        'faixa_cliente': 'Tier 1 - Parceiro Estratégico (Top 20%)',
        'share_global_faturamento_pct': 22.4,
    }
    mock_analitica_repo.consultar_avaliacoes_cooperativa.return_value = [
        {
            'avaliacao_id': str(uuid4()),
            'cooperativa_id': str(coop_id),
            'nome_cooperativa': 'Cooperativa Recicla Paulistana',
            'nome_empresa_avaliadora': 'Indústria Circular SA',
            'nota': 5,
            'comentario': 'Excelente separação e prensagem impecável.',
            'data_avaliacao': '2026-09-22T16:00:00',
        }
    ]

    mock_rec_repo = MagicMock()

    return {
        'cooperativa_id': coop_id,
        'empresa_id': emp_id,
        'venda_repository': mock_venda_repo,
        'analitica_repository': mock_analitica_repo,
        'recomendacao_repository': mock_rec_repo,
    }


def test_financeiro_novas_tools_com_perfil(fake_repositories):
    coop_id = fake_repositories['cooperativa_id']
    tk = FinanceiroToolkit(repositories=fake_repositories)
    config = {'configurable': {'cooperativa_id': str(coop_id)}}

    # 1. Faturamento de vendas
    res_vendas = tk.consultar_faturamento_vendas(config=config, mes_referencia='2026-09')
    assert res_vendas['status'] == 'ok'
    assert res_vendas['cooperativa_id'] == str(coop_id)
    assert res_vendas['faturamento_geral_reais'] == 4500.0
    assert len(res_vendas['vendas']) == 1

    # 2. Saude financeira ESG
    res_esg = tk.consultar_saude_financeira_esg(config=config, mes_referencia='2026-09')
    assert res_esg['status'] == 'ok'
    assert res_esg['total_meses_apurados'] == 1
    assert res_esg['historico_esg'][0]['valor_medio_per_capita_reais'] == 2083.33

    # 3. Curva ABC
    res_abc = tk.consultar_curva_abc_receita(config=config)
    assert res_abc['status'] == 'ok'
    assert len(res_abc['curva_abc']) == 1
    assert res_abc['curva_abc'][0]['classificacao_abc'] == 'Classe A (Alto Impacto / 80%)'


def test_financeiro_recorte_perfil_ausente_retorna_erro(fake_repositories):
    tk = FinanceiroToolkit(repositories=fake_repositories)
    config_vazio = {'configurable': {}}

    res = tk.consultar_faturamento_vendas(config=config_vazio)
    assert res['status'] == 'error'
    assert 'cooperativa_id ausente' in res['message']


def test_coleta_novas_tools_com_perfil(fake_repositories):
    coop_id = fake_repositories['cooperativa_id']
    tk = ColetaToolkit(repositories=fake_repositories)
    config = {'configurable': {'cooperativa_id': str(coop_id)}}

    # 1. Desempenho de equipes
    res_desempenho = tk.consultar_desempenho_equipes_triagem(config=config, mes_referencia='2026-09')
    assert res_desempenho['status'] == 'ok'
    assert res_desempenho['ranking_equipes'][0]['ranking_produtividade'] == 1
    assert res_desempenho['ranking_equipes'][0]['taxa_rejeito_equipe_pct'] == 2.86

    # 2. Anomalias de fluxo
    res_anomalias = tk.consultar_anomalias_fluxo_coleta(config=config)
    assert res_anomalias['status'] == 'ok'
    assert len(res_anomalias['eventos']) == 1
    assert res_anomalias['eventos'][0]['variacao_vs_anterior_kg'] == 330.0


def test_coleta_recorte_perfil_ausente_retorna_erro(fake_repositories):
    tk = ColetaToolkit(repositories=fake_repositories)
    config_vazio = {'configurable': {}}

    res = tk.consultar_desempenho_equipes_triagem(config=config_vazio)
    assert res['status'] == 'error'
    assert 'cooperativa_id ausente' in res['message']


def test_material_estoque_novas_tools_com_perfil(fake_repositories):
    coop_id = fake_repositories['cooperativa_id']
    tk = MaterialEstoqueToolkit(repositories=fake_repositories)
    config = {'configurable': {'cooperativa_id': str(coop_id)}}

    # 1. Cobertura e giro
    res_cobertura = tk.consultar_cobertura_giro_estoque(config=config)
    assert res_cobertura['status'] == 'ok'
    assert res_cobertura['cobertura_estoque'][0]['status_capacidade_armazenamento'] == 'Alto Volume / Atenção a Espaço Físico'

    # 2. Ranking de mercado
    res_mercado = tk.consultar_ranking_mercado_materiais(config=config, categoria_material='Latinha')
    assert res_mercado['status'] == 'ok'
    assert res_mercado['ranking_mercado'][0]['share_material_mes_pct'] == 34.5


def test_material_estoque_recorte_perfil_ausente_retorna_erro(fake_repositories):
    tk = MaterialEstoqueToolkit(repositories=fake_repositories)
    config_vazio = {'configurable': {}}

    res = tk.consultar_cobertura_giro_estoque(config=config_vazio)
    assert res['status'] == 'error'
    assert 'cooperativa_id ausente' in res['message']


def test_recomendacao_novas_tools_com_perfil(fake_repositories):
    emp_id = fake_repositories['empresa_id']
    tk = RecomendacaoToolkit(
        repository=fake_repositories['recomendacao_repository'],
        repositories=fake_repositories,
    )
    config = {'configurable': {'empresa_id': str(emp_id)}}

    # 1. Perfil RFV da empresa compradora
    res_rfv = tk.consultar_perfil_comprador_rfv(config=config)
    assert res_rfv['status'] == 'ok'
    assert res_rfv['perfil_rfv']['faixa_cliente'] == 'Tier 1 - Parceiro Estratégico (Top 20%)'
    assert res_rfv['perfil_rfv']['total_pedidos_concluidos'] == 15

    # 2. Avaliacoes qualitativas de cooperativa
    res_aval = tk.consultar_avaliacoes_cooperativa(config=config, cooperativa_nome='Recicla Paulistana')
    assert res_aval['status'] == 'ok'
    assert len(res_aval['avaliacoes']) == 1
    assert res_aval['avaliacoes'][0]['nota'] == 5


def test_recomendacao_rfv_perfil_ausente_retorna_erro(fake_repositories):
    tk = RecomendacaoToolkit(
        repository=fake_repositories['recomendacao_repository'],
        repositories=fake_repositories,
    )
    config_vazio = {'configurable': {}}

    res = tk.consultar_perfil_comprador_rfv(config=config_vazio)
    assert res['status'] == 'error'
    assert 'empresa_id ausente' in res['message']
