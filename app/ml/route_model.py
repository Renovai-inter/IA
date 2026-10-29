import csv
import logging
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

from app.ml.sp_zones import (
    DATA_DIR,
    extrair_features_trecho,
    inferir_coordenadas_sp,
)

log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / 'sp_route_model.joblib'
DEFAULT_DATASET_PATH = DATA_DIR / 'sp_rotas_treinamento.csv'

FEATURE_COLUMNS = (
    'distancia_haversine_km',
    'distancia_viaria_km',
    'codigo_zona_origem',
    'codigo_zona_destino',
    'mudanca_zona',
    'fator_retencao',
    'fator_horario_sp',
    'hora_saida',
    'peso_estimado_kg',
)
TARGET_COLUMN = 'tempo_total_min'


class PontoParada(BaseModel):
    id: str = Field(default='', description='Identificador do ponto de coleta ou parada.')
    endereco: str = Field(..., description='Endereço textual da parada em São Paulo.')
    bairro: Optional[str] = Field(default=None, description='Bairro em São Paulo.')
    cep: Optional[str] = Field(default=None, description='CEP da parada (01000-000 a 08999-999).')
    latitude: Optional[float] = Field(default=None, description='Latitude opcional.')
    longitude: Optional[float] = Field(default=None, description='Longitude opcional.')
    peso_estimado_kg: float = Field(default=200.0, description='Volume estimado de coleta em kg.')


class ParadaOtimizada(BaseModel):
    ordem: int
    id: str
    endereco: str
    zona: str
    latitude: float
    longitude: float
    distancia_trecho_km: float
    tempo_trecho_min: float
    tempo_acumulado_min: float


class ResultadoRotaOtimizada(BaseModel):
    origem: str
    hora_saida: int
    paradas_ordenadas: List[ParadaOtimizada]
    distancia_total_km: float
    tempo_total_estimado_min: float
    tempo_ordem_original_min: float
    economia_estimada_min: float
    economia_percentual: float


def carregar_dataset_csv(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> tuple[list[list[float]], list[float]]:
    """Lê o dataset tabular de trechos de rotas em São Paulo a partir de um arquivo CSV em IA/data/."""
    caminho = Path(dataset_path)
    X: list[list[float]] = []
    y: list[float] = []

    with open(caminho, mode='r', encoding='utf-8') as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            X.append([float(linha[col]) for col in FEATURE_COLUMNS])
            y.append(float(linha[TARGET_COLUMN]))

    return X, y


class ModeloRotaSupervisionadoSP:
    """Modelo supervisionado scikit-learn para predição de tempo de deslocamento entre paradas
    em São Paulo e otimização combinatória da rota (Nearest Neighbor + 2-Opt)."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
    ):
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.pipeline: Optional[Any] = None

    def _criar_pipeline(self) -> 'Pipeline':
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.9,
                random_state=42,
            )),
        ])

    def treinar(
        self,
        dataset_path: Optional[Path | str] = None,
        salvar_artefato: bool = True,
    ) -> dict[str, float]:
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split

        caminho_csv = Path(dataset_path) if dataset_path else self.dataset_path
        X, y = carregar_dataset_csv(caminho_csv)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.pipeline = self._criar_pipeline()
        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        if salvar_artefato:
            self.salvar()

        return {
            'mae_min': round(mae, 3),
            'r2_score': round(r2, 4),
            'amostras_treino': len(X_train),
            'amostras_teste': len(X_test),
        }

    def treinar_com_neo4j(
        self,
        rota_repository,
        incluir_csv_base: bool = True,
        apenas_reais: bool = False,
        salvar_artefato: bool = True,
    ) -> dict[str, float | int]:
        """Retreina o pipeline supervisionado utilizando os trechos armazenados no Neo4j
        (combinados com o dataset CSV base em IA/data/ quando incluir_csv_base=True)."""
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split

        X_neo4j, y_neo4j = rota_repository.buscar_trechos_treinamento(apenas_reais=apenas_reais)

        X_csv: list[list[float]] = []
        y_csv: list[float] = []
        if incluir_csv_base or len(X_neo4j) < 20:
            X_csv, y_csv = carregar_dataset_csv(self.dataset_path)

        X = X_csv + X_neo4j
        y = y_csv + y_neo4j

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.pipeline = self._criar_pipeline()
        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        if salvar_artefato:
            self.salvar()

        return {
            'mae_min': round(mae, 3),
            'r2_score': round(r2, 4),
            'amostras_neo4j': len(X_neo4j),
            'amostras_csv': len(X_csv),
            'amostras_treino': len(X_train),
            'amostras_teste': len(X_test),
        }

    def salvar(self, caminho: Optional[Path | str] = None) -> Path:
        import joblib

        destino = Path(caminho) if caminho else self.model_path
        destino.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, destino)
        return destino

    def carregar(self, caminho: Optional[Path | str] = None) -> 'Pipeline':
        import joblib

        origem = Path(caminho) if caminho else self.model_path
        self.pipeline = joblib.load(origem)
        return self.pipeline

    def garantir_modelo(self) -> 'Pipeline':
        if self.pipeline is not None:
            return self.pipeline
        if self.model_path.exists():
            return self.carregar()
        log.info('Artefato de rota SP não encontrado em %s; treinando a partir de %s.', self.model_path, self.dataset_path)
        self.treinar(salvar_artefato=True)
        return self.pipeline

    def prever_trecho(
        self,
        lat_origem: float,
        lon_origem: float,
        zona_origem: str,
        lat_destino: float,
        lon_destino: float,
        zona_destino: str,
        hora_saida: int = 9,
        peso_estimado_kg: float = 200.0,
    ) -> tuple[float, float]:
        """Retorna (distancia_viaria_km, tempo_predito_min) para um trecho."""
        modelo = self.garantir_modelo()
        feats = extrair_features_trecho(
            lat_origem=lat_origem,
            lon_origem=lon_origem,
            zona_origem=zona_origem,
            lat_destino=lat_destino,
            lon_destino=lon_destino,
            zona_destino=zona_destino,
            hora_saida=hora_saida,
            peso_estimado_kg=peso_estimado_kg,
        )
        dist_viaria_km = feats[1]
        if dist_viaria_km < 0.01:
            return 0.0, 0.0
        tempo_min = float(modelo.predict([feats])[0])
        return round(dist_viaria_km, 2), round(max(1.0, tempo_min), 2)

    @staticmethod
    def _custo_sequencia(ordem: list[int], matriz_tempo: list[list[float]]) -> float:
        custo = 0.0
        atual = 0
        for idx in ordem:
            custo += matriz_tempo[atual][idx]
            atual = idx
        return custo

    @classmethod
    def _resolver_tsp_2opt(cls, n_paradas: int, matriz_tempo: list[list[float]]) -> list[int]:
        """Encontra a sequência ótima de paradas partindo do índice 0 (cooperativa)
        via Vizinho Mais Próximo + refinamento local 2-Opt."""
        nao_visitados = set(range(1, n_paradas + 1))
        rota: list[int] = []
        atual = 0

        while nao_visitados:
            proximo = min(nao_visitados, key=lambda j: matriz_tempo[atual][j])
            rota.append(proximo)
            nao_visitados.remove(proximo)
            atual = proximo

        melhorou = True
        while melhorou and len(rota) >= 3:
            melhorou = False
            melhor_custo = cls._custo_sequencia(rota, matriz_tempo)
            for i in range(len(rota) - 1):
                for k in range(i + 1, len(rota)):
                    nova_rota = rota[:i] + list(reversed(rota[i:k + 1])) + rota[k + 1:]
                    novo_custo = cls._custo_sequencia(nova_rota, matriz_tempo)
                    if novo_custo + 1e-6 < melhor_custo:
                        rota = nova_rota
                        melhor_custo = novo_custo
                        melhorou = True
        return rota

    def otimizar_rota(
        self,
        origem: PontoParada,
        paradas: List[PontoParada],
        hora_saida: int = 9,
    ) -> ResultadoRotaOtimizada:
        if not paradas:
            return ResultadoRotaOtimizada(
                origem=origem.endereco,
                hora_saida=hora_saida,
                paradas_ordenadas=[],
                distancia_total_km=0.0,
                tempo_total_estimado_min=0.0,
                tempo_ordem_original_min=0.0,
                economia_estimada_min=0.0,
                economia_percentual=0.0,
            )

        todos_pontos = [origem] + list(paradas)
        coords_zonas = [
            inferir_coordenadas_sp(
                endereco=p.endereco,
                bairro=p.bairro,
                cep=p.cep,
                latitude=p.latitude,
                longitude=p.longitude,
            )
            for p in todos_pontos
        ]

        total = len(todos_pontos)
        matriz_dist = [[0.0] * total for _ in range(total)]
        matriz_tempo = [[0.0] * total for _ in range(total)]

        for i in range(total):
            lat_i, lon_i, zona_i = coords_zonas[i]
            for j in range(total):
                if i == j:
                    continue
                lat_j, lon_j, zona_j = coords_zonas[j]
                dist_ij, tempo_ij = self.prever_trecho(
                    lat_origem=lat_i,
                    lon_origem=lon_i,
                    zona_origem=zona_i,
                    lat_destino=lat_j,
                    lon_destino=lon_j,
                    zona_destino=zona_j,
                    hora_saida=hora_saida,
                    peso_estimado_kg=todos_pontos[j].peso_estimado_kg,
                )
                matriz_dist[i][j] = dist_ij
                matriz_tempo[i][j] = tempo_ij

        ordem_original = list(range(1, total))
        tempo_original = self._custo_sequencia(ordem_original, matriz_tempo)

        ordem_otima = self._resolver_tsp_2opt(len(paradas), matriz_tempo)

        paradas_ordenadas: List[ParadaOtimizada] = []
        tempo_acum = 0.0
        dist_acum = 0.0
        anterior = 0

        for pos, idx_ponto in enumerate(ordem_otima, start=1):
            ponto = todos_pontos[idx_ponto]
            lat_p, lon_p, zona_p = coords_zonas[idx_ponto]
            d_trecho = matriz_dist[anterior][idx_ponto]
            t_trecho = matriz_tempo[anterior][idx_ponto]

            dist_acum += d_trecho
            tempo_acum += t_trecho

            paradas_ordenadas.append(
                ParadaOtimizada(
                    ordem=pos,
                    id=ponto.id or f'parada-{pos}',
                    endereco=ponto.endereco,
                    zona=zona_p,
                    latitude=lat_p,
                    longitude=lon_p,
                    distancia_trecho_km=round(d_trecho, 2),
                    tempo_trecho_min=round(t_trecho, 2),
                    tempo_acumulado_min=round(tempo_acum, 2),
                )
            )
            anterior = idx_ponto

        economia_min = max(0.0, tempo_original - tempo_acum)
        economia_pct = (economia_min / tempo_original * 100.0) if tempo_original > 0 else 0.0

        return ResultadoRotaOtimizada(
            origem=origem.endereco,
            hora_saida=hora_saida,
            paradas_ordenadas=paradas_ordenadas,
            distancia_total_km=round(dist_acum, 2),
            tempo_total_estimado_min=round(tempo_acum, 2),
            tempo_ordem_original_min=round(tempo_original, 2),
            economia_estimada_min=round(economia_min, 2),
            economia_percentual=round(economia_pct, 1),
        )
