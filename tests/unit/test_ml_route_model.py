from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.ml.route_model import (
    DEFAULT_DATASET_PATH,
    ModeloRotaSupervisionadoSP,
    PontoParada,
    carregar_dataset_csv,
)
from app.ml.sp_zones import (
    classificar_zona_por_cep,
    extrair_features_trecho,
    fator_horario_sp,
    inferir_coordenadas_sp,
)
from app.repository.neo4j.rota_repository import RotaRepository


def test_classificar_zona_por_cep_sp():
    assert classificar_zona_por_cep('01310-100') == 'CENTRO'
    assert classificar_zona_por_cep('02011-000') == 'NORTE'
    assert classificar_zona_por_cep('03306-000') == 'LESTE'
    assert classificar_zona_por_cep('04015-000') == 'SUL'
    assert classificar_zona_por_cep('05422-000') == 'OESTE'


def test_extrair_features_trecho_reflete_horario_de_pico():
    lat_o, lon_o, zona_o = inferir_coordenadas_sp(bairro='Tatuapé')
    lat_d, lon_d, zona_d = inferir_coordenadas_sp(bairro='Pinheiros')

    feats_madrugada = extrair_features_trecho(
        lat_o, lon_o, zona_o, lat_d, lon_d, zona_d, hora_saida=5
    )
    feats_pico = extrair_features_trecho(
        lat_o, lon_o, zona_o, lat_d, lon_d, zona_d, hora_saida=18
    )

    assert feats_pico[0] > 5.0
    assert feats_pico[4] == 1.0  # mudança LESTE -> OESTE
    assert feats_pico[6] > feats_madrugada[6]
    assert fator_horario_sp(18) > fator_horario_sp(5)


def test_modelo_rota_supervisionado_treina_de_csv_e_otimiza(tmp_path: Path):
    assert DEFAULT_DATASET_PATH.exists()
    X, y = carregar_dataset_csv(DEFAULT_DATASET_PATH)
    assert len(X) >= 1000
    assert len(X) == len(y)

    caminho_modelo = tmp_path / 'sp_route_test.joblib'
    modelo = ModeloRotaSupervisionadoSP(
        model_path=caminho_modelo,
        dataset_path=DEFAULT_DATASET_PATH,
    )

    metricas = modelo.treinar(salvar_artefato=True)
    assert caminho_modelo.exists()
    assert metricas['r2_score'] > 0.85
    assert metricas['mae_min'] < 5.0

    # Sequência propositalmente fora de ordem (zigue-zague Leste -> Oeste -> Leste -> Centro)
    origem = PontoParada(id='coop-1', endereco='Rua Tuiuti, 1200 - Tatuapé', bairro='Tatuapé', cep='03307-000')
    paradas_desordenadas = [
        PontoParada(id='p-pinheiros', endereco='Rua dos Pinheiros, 500', bairro='Pinheiros', cep='05422-000', peso_estimado_kg=300.0),
        PontoParada(id='p-mooca',     endereco='Rua da Mooca, 1800',     bairro='Mooca',     cep='03104-000', peso_estimado_kg=250.0),
        PontoParada(id='p-se',        endereco='Praça da Sé, 100',       bairro='Sé',        cep='01001-000', peso_estimado_kg=200.0),
    ]

    resultado = modelo.otimizar_rota(origem=origem, paradas=paradas_desordenadas, hora_saida=8)

    assert len(resultado.paradas_ordenadas) == 3
    assert resultado.paradas_ordenadas[0].id == 'p-mooca'
    assert resultado.tempo_total_estimado_min <= resultado.tempo_ordem_original_min
    assert resultado.economia_estimada_min >= 0.0


def test_rota_repository_salva_e_extrai_trechos_para_retreino(tmp_path: Path):
    driver = MagicMock()
    sessao = MagicMock()
    driver.session.return_value.__enter__.return_value = sessao

    sessao.run.return_value.single.return_value = {
        'rota_id': 'rota-sp-001',
        'paradas_salvas': 2,
        'atualizados': 2,
    }
    sessao.run.return_value.__iter__.return_value = iter([
        {
            'distancia_haversine_km': 4.2,
            'distancia_viaria_km': 5.8,
            'codigo_zona_origem': 3.0,
            'codigo_zona_destino': 3.0,
            'mudanca_zona': 0.0,
            'fator_retencao': 1.05,
            'fator_horario_sp': 1.55,
            'hora_saida': 8.0,
            'peso_estimado_kg': 250.0,
            'tempo_total_min': 24.5,
        },
        {
            'distancia_haversine_km': 6.1,
            'distancia_viaria_km': 8.4,
            'codigo_zona_origem': 3.0,
            'codigo_zona_destino': 0.0,
            'mudanca_zona': 1.0,
            'fator_retencao': 1.40,
            'fator_horario_sp': 1.55,
            'hora_saida': 8.0,
            'peso_estimado_kg': 300.0,
            'tempo_total_min': 39.0,
        },
    ])

    repo = RotaRepository(db=driver, database='neo4j')
    modelo = ModeloRotaSupervisionadoSP(
        model_path=tmp_path / 'sp_route_neo4j.joblib',
        dataset_path=DEFAULT_DATASET_PATH,
    )

    origem = PontoParada(id='coop-1', endereco='Rua Tuiuti, 1200', bairro='Tatuapé', cep='03307-000')
    paradas = [
        PontoParada(id='p1', endereco='Rua da Mooca, 1800', bairro='Mooca', cep='03104-000', peso_estimado_kg=250.0),
        PontoParada(id='p2', endereco='Praça da Sé, 100', bairro='Sé', cep='01001-000', peso_estimado_kg=300.0),
    ]
    resultado = modelo.otimizar_rota(origem=origem, paradas=paradas, hora_saida=8)

    res_salvar = repo.salvar_rota_otimizada(
        resultado=resultado,
        origem=origem,
        paradas_entrada=paradas,
        cooperativa_id='coop-1',
        rota_id='rota-sp-001',
    )
    assert res_salvar['rota_id'] == 'rota-sp-001'
    assert res_salvar['paradas_salvas'] == 2

    atualizados = repo.registrar_tempos_reais('rota-sp-001', {'p1': 24.5, 'p2': 39.0})
    assert atualizados == 2

    metricas = modelo.treinar_com_neo4j(
        rota_repository=repo,
        incluir_csv_base=True,
        apenas_reais=True,
        salvar_artefato=True,
    )
    assert metricas['amostras_neo4j'] == 2
    assert metricas['amostras_csv'] >= 1000
    assert metricas['mae_min'] < 5.0


def test_endpoints_rotas_otimizar_e_retreinar(tmp_path: Path):
    driver = MagicMock()
    sessao = MagicMock()
    driver.session.return_value.__enter__.return_value = sessao
    sessao.run.return_value.single.return_value = {
        'rota_id': 'rota-api-123',
        'paradas_salvas': 2,
        'atualizados': 2,
    }
    sessao.run.return_value.__iter__.return_value = iter([
        {
            'distancia_haversine_km': 4.2,
            'distancia_viaria_km': 5.8,
            'codigo_zona_origem': 3.0,
            'codigo_zona_destino': 3.0,
            'mudanca_zona': 0.0,
            'fator_retencao': 1.05,
            'fator_horario_sp': 1.55,
            'hora_saida': 9.0,
            'peso_estimado_kg': 200.0,
            'tempo_total_min': 21.0,
        }
    ])

    repo = RotaRepository(db=driver, database='neo4j')
    modelo = ModeloRotaSupervisionadoSP(
        model_path=tmp_path / 'sp_route_api.joblib',
        dataset_path=DEFAULT_DATASET_PATH,
    )
    app.state.container = SimpleNamespace(
        modelo_rotas=modelo,
        repositories={'rota_repository': repo},
    )

    client = TestClient(app, raise_server_exceptions=True)

    resp_otim = client.post(
        '/rotas/otimizar',
        json={
            'cooperativa_id': 'coop-1',
            'rota_id': 'rota-api-123',
            'nome_rota': 'Rota Coleta Leste-Centro',
            'hora_saida': 9,
            'salvar_no_neo4j': True,
            'origem': {
                'id': 'coop-1',
                'endereco': 'Rua Tuiuti, 1200',
                'bairro': 'Tatuapé',
                'cep': '03307-000',
            },
            'paradas': [
                {
                    'id': 'p-se',
                    'endereco': 'Praça da Sé, 100',
                    'bairro': 'Sé',
                    'cep': '01001-000',
                    'peso_estimado_kg': 220.0,
                },
                {
                    'id': 'p-mooca',
                    'endereco': 'Rua da Mooca, 1800',
                    'bairro': 'Mooca',
                    'cep': '03104-000',
                    'peso_estimado_kg': 180.0,
                },
            ],
        },
    )
    assert resp_otim.status_code == 200
    dados_otim = resp_otim.json()
    assert dados_otim['salvo_no_neo4j'] is True
    assert dados_otim['rota_id'] == 'rota-api-123'
    assert dados_otim['paradas_ordenadas'][0]['id'] == 'p-mooca'

    resp_retreino = client.post(
        '/rotas/retreinar',
        json={
            'rota_id': 'rota-api-123',
            'tempos_reais_min': {'p-mooca': 19.5, 'p-se': 26.0},
            'incluir_csv_base': True,
            'apenas_reais': True,
        },
    )
    assert resp_retreino.status_code == 200
    dados_retreino = resp_retreino.json()
    assert dados_retreino['status'] == 'ok'
    assert dados_retreino['tempos_reais_atualizados'] == 2
    assert dados_retreino['amostras_neo4j'] == 1


def test_logistica_agent_no_deterministico_sem_llm(tmp_path: Path):
    import json
    from app.agents.logistica_agent import LogisticaAgent

    modelo = ModeloRotaSupervisionadoSP(
        model_path=tmp_path / 'sp_route_node.joblib',
        dataset_path=DEFAULT_DATASET_PATH,
    )
    repo = MagicMock()
    repo.salvar_rota_otimizada.return_value = {'rota_id': 'rota-node-1', 'paradas_salvas': 3}

    agente = LogisticaAgent(modelo_rotas=modelo, rota_repository=repo)
    assert agente.llm is None

    state = {
        'messages': [
            {
                'role': 'user',
                'content': 'Qual a melhor rota saindo do Tatuapé às 18h passando por Pinheiros, Mooca e Sé?',
            }
        ],
        'perfil_ctx': SimpleNamespace(cooperativa_id='coop-1'),
    }
    saida = agente.run(state)

    assert saida['agentes_chamados'] == ['logistica']
    payload = json.loads(saida['especialista_outputs']['logistica'].conteudo)
    assert payload['dominio'] == 'logistica'
    assert payload['intencao'] == 'otimizar_rota'
    assert payload['indicadores']['hora_saida'] == 18
    assert payload['indicadores']['qtd_paradas'] == 3
    assert payload['indicadores']['salvo_no_neo4j'] is True
    assert '1º Mooca' in payload['resposta']


def test_logistica_agent_enderecos_livres_aciona_modelo_supervisionado(tmp_path: Path):
    import json
    from app.agents.logistica_agent import LogisticaAgent

    modelo = ModeloRotaSupervisionadoSP(
        model_path=tmp_path / 'sp_route_free_addr.joblib',
        dataset_path=DEFAULT_DATASET_PATH,
    )
    repo = MagicMock()
    repo.buscar_dados_rota_cooperativa.return_value = None
    repo.salvar_rota_otimizada.return_value = {'rota_id': 'rota-free-1', 'paradas_salvas': 2}

    agente = LogisticaAgent(modelo_rotas=modelo, rota_repository=repo)

    # Caso 1: 3 endereços livres sem frase padrão ("Saindo de...") e sem bairros do CSV
    state_1 = {
        'messages': [
            {
                'role': 'user',
                'content': 'Preciso montar uma rota passando pela Av. Paulista, Rua Vergueiro e Largo da Batata',
            }
        ],
        'perfil_ctx': SimpleNamespace(cooperativa_id='coop-1'),
    }
    saida_1 = agente.run(state_1)
    payload_1 = json.loads(saida_1['especialista_outputs']['logistica'].conteudo)

    assert 'esclarecer' not in payload_1
    assert payload_1['indicadores']['qtd_paradas'] == 2
    assert payload_1['indicadores']['distancia_total_km'] > 0
    assert payload_1['indicadores']['tempo_total_estimado_min'] > 0

    # Caso 2: 3 logradouros arbitrários com números de imóvel (fora do catálogo fixo)
    state_2 = {
        'messages': [
            {
                'role': 'user',
                'content': (
                    'Monte a rota passando pela Rua dos Três Irmãos, 200, '
                    'Avenida Professor Alfonso Bovero, 500 e Rua Sergipe, 400'
                ),
            }
        ],
        'perfil_ctx': SimpleNamespace(cooperativa_id='coop-1'),
    }
    saida_2 = agente.run(state_2)
    payload_2 = json.loads(saida_2['especialista_outputs']['logistica'].conteudo)

    assert 'esclarecer' not in payload_2
    assert payload_2['indicadores']['qtd_paradas'] == 2
    assert payload_2['indicadores']['distancia_total_km'] > 0
    assert payload_2['indicadores']['tempo_total_estimado_min'] > 0


