import csv
import random
from pathlib import Path

from app.ml.route_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ModeloRotaSupervisionadoSP,
)
from app.ml.sp_zones import BAIRROS_COORDS_SP, extrair_features_trecho


def exportar_dataset_csv(
    destino_csv: Path = DEFAULT_DATASET_PATH,
    n_amostras: int = 1500,
    seed: int = 42,
) -> Path:
    """Gera o arquivo CSV de treinamento em IA/data/sp_rotas_treinamento.csv calibrado
    pelas velocidades médias da CET-SP por zona/faixa horária e tempo de coleta."""
    rng = random.Random(seed)
    bairros = list(BAIRROS_COORDS_SP.values())
    destino_csv.parent.mkdir(parents=True, exist_ok=True)

    cabecalho = list(FEATURE_COLUMNS) + [TARGET_COLUMN]

    with open(destino_csv, mode='w', encoding='utf-8', newline='') as f:
        escritor = csv.writer(f)
        escritor.writerow(cabecalho)

        for _ in range(n_amostras):
            lat_o, lon_o, zona_o = rng.choice(bairros)
            lat_d, lon_d, zona_d = rng.choice(bairros)

            lat_o += rng.uniform(-0.018, 0.018)
            lon_o += rng.uniform(-0.018, 0.018)
            lat_d += rng.uniform(-0.018, 0.018)
            lon_d += rng.uniform(-0.018, 0.018)

            hora = rng.choice([6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20])
            peso_kg = round(rng.uniform(50.0, 1500.0), 1)

            feats = extrair_features_trecho(
                lat_origem=lat_o,
                lon_origem=lon_o,
                zona_origem=zona_o,
                lat_destino=lat_d,
                lon_destino=lon_d,
                zona_destino=zona_d,
                hora_saida=hora,
                peso_estimado_kg=peso_kg,
            )

            dist_viaria = feats[1]
            mudanca_zona = feats[4]
            retencao = feats[5]
            mult_hora = feats[6]

            velocidade_base_kmh = 26.0 / (retencao * mult_hora)
            tempo_deslocamento_min = (dist_viaria / max(velocidade_base_kmh, 8.0)) * 60.0
            penalidade_semaforos_min = dist_viaria * 0.85 + (4.5 if mudanca_zona > 0 else 1.2)
            tempo_carregamento_min = 3.0 + (peso_kg / 350.0)
            ruido = rng.gauss(0.0, 0.6)

            tempo_total_min = round(
                max(2.5, tempo_deslocamento_min + penalidade_semaforos_min + tempo_carregamento_min + ruido),
                2,
            )
            escritor.writerow(feats + [tempo_total_min])

    return destino_csv


def main() -> None:
    if not DEFAULT_DATASET_PATH.exists():
        exportar_dataset_csv(DEFAULT_DATASET_PATH)

    modelo = ModeloRotaSupervisionadoSP(
        model_path=DEFAULT_MODEL_PATH,
        dataset_path=DEFAULT_DATASET_PATH,
    )
    metricas = modelo.treinar(salvar_artefato=True)
    print(
        f"Dataset lido de '{DEFAULT_DATASET_PATH}' | "
        f"Modelo salvo em '{DEFAULT_MODEL_PATH}' | "
        f"MAE={metricas['mae_min']} min | R2={metricas['r2_score']}"
    )


if __name__ == '__main__':
    main()
