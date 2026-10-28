from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.repository.neo4j.recomendacao_repository import RecomendacaoRepository
from app.tools.base import Toolkit


class RecomendarCooperativasArgs(BaseModel):
    material: str = Field(
        ...,
        description="Nome ou categoria do material reciclável desejado (ex: 'PET', 'Alumínio', 'Papelão')."
    )
    raio_max_km: float = Field(
        default=40.0,
        description='Distância máxima em km na malha viária entre a empresa compradora e a cooperativa.'
    )
    volume_minimo_kg: float = Field(
        default=500.0,
        description='Quantidade mínima desejada do material em kg.'
    )
    empresa_id: Optional[str] = Field(
        default=None,
        description='Identificador opcional da empresa compradora para cálculo de distância e favoritas.'
    )


class RecomendacaoToolkit(Toolkit):
    nome = 'recomendacao_toolkit'
    descricao = 'Tools de recomendação multicritério de cooperativas no Neo4j.'
    repositories = {
        'recomendacao_repository': RecomendacaoRepository,
    }

    def __init__(self, repository: RecomendacaoRepository):
        self.repository = repository

    @Toolkit.timed(categoria='tool', nome='recomendar_cooperativas')
    def recomendar_cooperativas(
        self,
        material: str,
        raio_max_km: float = 40.0,
        volume_minimo_kg: float = 500.0,
        empresa_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """Consulta o grafo Neo4j e recomenda as cooperativas ideais para aquisição de determinado
        material reciclável com base em estoque imediato disponível, menor distância viária,
        capacidade operacional, conformidade e fontes validadas."""
        Toolkit.log.debug('acessou tool - recomendar_cooperativas')

        if not material or not material.strip():
            return {'status': 'error', 'message': 'Material não informado para busca de cooperativas.'}

        if not empresa_id and config:
            cfg = (config or {}).get('configurable') or {}
            if cfg.get('empresa_id'):
                empresa_id = str(cfg['empresa_id'])

        recomendacoes = self.repository.recomendar_cooperativas(
            material=material,
            empresa_id=empresa_id,
            raio_max_km=raio_max_km,
            volume_minimo_kg=volume_minimo_kg,
        )

        if not recomendacoes:
            return {'status': 'error', 'message': 'Nenhuma cooperativa compatível foi encontrada para os filtros informados.'}

        itens = [
            {
                'cooperativa_id':       r.cooperativa_id,
                'cooperativa_nome':     r.cooperativa_nome,
                'material_disponivel':  r.material,
                'estoque_atual_kg':     r.estoque_atual_kg,
                'capacidade_mensal_kg': r.capacidade_mensal_kg,
                'preco_sugerido_medio': r.preco_sugerido_medio,
                'distancia_km':         r.distancia_km,
                'score_conformidade':   r.score_conformidade,
                'fontes_validadas':     r.fontes_validadas,
                'favorita':             r.favorita,
                'score_final':          r.score_final,
            }
            for r in recomendacoes
        ]

        return {
            'status': 'ok',
            'count': len(itens),
            'itens': itens,
        }

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.recomendar_cooperativas,
                name='recomendar_cooperativas',
                description=self.recomendar_cooperativas.__doc__,
                args_schema=RecomendarCooperativasArgs,
            ),
        ]
