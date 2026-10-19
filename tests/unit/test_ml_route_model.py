from pathlib import Path

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
