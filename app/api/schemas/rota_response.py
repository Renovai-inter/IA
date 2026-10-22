from typing import List, Optional

from pydantic import BaseModel

from app.ml.route_model import ParadaOtimizada


class OtimizarRotaResponse(BaseModel):
    """Resposta do endpoint POST /rotas/otimizar."""
    rota_id: Optional[str] = None
    origem: str
    hora_saida: int
    paradas_ordenadas: List[ParadaOtimizada]
    distancia_total_km: float
    tempo_total_estimado_min: float
    tempo_ordem_original_min: float
    economia_estimada_min: float
    economia_percentual: float
    salvo_no_neo4j: bool = False
    paradas_salvas_neo4j: int = 0


class RetreinarRotaResponse(BaseModel):
    """Resposta do endpoint POST /rotas/retreinar."""
    status: str
    tempos_reais_atualizados: int = 0
    mae_min: float
    r2_score: float
    amostras_neo4j: int
    amostras_csv: int
    amostras_treino: int
    amostras_teste: int
