from unittest.mock import MagicMock

from app.repository.neo4j.recomendacao_repository import (
    RecomendacaoRepository,
    RecomendacaoCooperativa,
    calcular_score_multicriterio,
)
from app.tools.recomendacao_tools import RecomendacaoToolkit


def test_calcular_score_multicriterio_pondera_estoque_e_distancia():
    score_otimo = calcular_score_multicriterio(
        estoque_atual_kg=2500.0,
        volume_minimo_kg=2000.0,
        distancia_km=8.0,
        raio_max_km=40.0,
        capacidade_mensal_kg=10000.0,
        score_conformidade=0.95,
        fontes_validadas=5,
        favorita=True,
    )
    score_baixo_estoque_longe = calcular_score_multicriterio(
        estoque_atual_kg=400.0,
        volume_minimo_kg=2000.0,
        distancia_km=35.0,
        raio_max_km=40.0,
        capacidade_mensal_kg=2000.0,
        score_conformidade=0.80,
        fontes_validadas=1,
        favorita=False,
    )
    assert score_otimo > score_baixo_estoque_longe
    assert 0.85 <= score_otimo <= 1.0
    assert 0.0 <= score_baixo_estoque_longe <= 0.45


def test_recomendacao_repository_ranqueia_cooperativas_do_neo4j():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_session.run.return_value = [
        {
            'cooperativa_id': 'coop-sp-20',
            'cooperativa_nome': 'Coop Leste Recicla',
            'material_nome': 'PEAD Fardo',
            'capacidade_mensal_kg': 6000.0,
            'estoque_atual_kg': 1100.0,
            'preco_sugerido_medio': 3.95,
            'distancia_km': 28.0,
            'score_conformidade': 0.85,
            'fontes_validadas': 2,
            'favorita': False,
        },
        {
            'cooperativa_id': 'coop-sp-10',
            'cooperativa_nome': 'Coop Verde Sul',
            'material_nome': 'PEAD Moído',
            'capacidade_mensal_kg': 12000.0,
            'estoque_atual_kg': 4500.0,
            'preco_sugerido_medio': 4.20,
            'distancia_km': 9.5,
            'score_conformidade': 0.96,
            'fontes_validadas': 5,
            'favorita': True,
        },
    ]

    repo = RecomendacaoRepository(db=mock_driver, database='neo4j')
    res = repo.recomendar_cooperativas(
        material='PEAD',
        empresa_id='emp-01',
        raio_max_km=40.0,
        volume_minimo_kg=2000.0,
    )

    assert len(res) == 2
    assert isinstance(res[0], RecomendacaoCooperativa)
    assert res[0].cooperativa_id == 'coop-sp-10'
    assert res[0].favorita is True
    assert res[0].estoque_atual_kg == 4500.0
    assert res[0].preco_sugerido_medio == 4.20
    assert res[0].score_final > res[1].score_final


def test_recomendacao_toolkit_retorna_status_ok_quando_encontra_cooperativas():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_session.run.return_value = [
        {
            'cooperativa_id': 'coop-sp-01',
            'cooperativa_nome': 'CoopMetais Brasil',
            'material_nome': 'Latinhas de Alumínio',
            'capacidade_mensal_kg': 18000.0,
            'estoque_atual_kg': 6500.0,
            'preco_sugerido_medio': 8.90,
            'distancia_km': 9.2,
            'score_conformidade': 0.98,
            'fontes_validadas': 5,
            'favorita': True,
        },
    ]

    repo = RecomendacaoRepository(db=mock_driver)
    toolkit = RecomendacaoToolkit(repository=repo)
    tool = toolkit.get_tools()[0]

    assert tool.name == 'recomendar_cooperativas'

    saida = tool.invoke(
        {'material': 'Alumínio', 'raio_max_km': 30.0, 'volume_minimo_kg': 500.0}
    )
    assert saida['status'] == 'ok'
    assert saida['count'] == 1
    assert saida['itens'][0]['cooperativa_id'] == 'coop-sp-01'
    assert saida['itens'][0]['estoque_atual_kg'] == 6500.0


def test_recomendacao_toolkit_retorna_status_error_quando_sem_resultados():
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = []

    repo = RecomendacaoRepository(db=mock_driver)
    toolkit = RecomendacaoToolkit(repository=repo)
    tool = toolkit.get_tools()[0]

    saida = tool.invoke(
        {'material': 'Vidro Temperado', 'raio_max_km': 10.0, 'volume_minimo_kg': 5000.0}
    )
    assert saida['status'] == 'error'
    assert 'Nenhuma cooperativa' in saida['message']
