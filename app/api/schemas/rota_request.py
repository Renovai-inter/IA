from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.ml.route_model import PontoParada


class OtimizarRotaRequest(BaseModel):
    """Payload enviado para otimizar uma rota de coleta e persistir no Neo4j."""
    cooperativa_id: str = Field(default='', examples=['coop-001'])
    rota_id: Optional[str] = Field(default=None, examples=['rota-zona-leste-01'])
    nome_rota: str = Field(default='Rota Otimizada IA', examples=['Coleta Manhã Zona Leste'])
    origem: PontoParada
    paradas: List[PontoParada] = Field(..., min_length=1)
    hora_saida: int = Field(default=9, ge=0, le=23)
    salvar_no_neo4j: bool = Field(default=True)


class RetreinarRotaRequest(BaseModel):
    """Payload opcional para registrar tempos reais de uma rota no Neo4j e retreinar o modelo."""
    rota_id: Optional[str] = Field(default=None, examples=['rota-zona-leste-01'])
    tempos_reais_min: Optional[Dict[str, float]] = Field(
        default=None,
        description='Mapa {ponto_id_ou_ordem: tempo_real_min} observado na execução da rota.',
    )
    incluir_csv_base: bool = Field(default=True)
    apenas_reais: bool = Field(default=False)
