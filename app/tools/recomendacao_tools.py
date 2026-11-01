from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from uuid import UUID
from app.repository.neo4j.recomendacao_repository import RecomendacaoRepository
from app.repository.postgresql.analitica_repository import AnaliticaRepository
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



class ConsultarPerfilCompradorRfvArgs(BaseModel):
    empresa_id: Optional[str] = Field(
        default=None,
        description='Identificador UUID da empresa compradora. Se ausente, lido do contexto da sessão.'
    )


class ConsultarAvaliacoesCooperativaArgs(BaseModel):
    cooperativa_nome: Optional[str] = Field(
        default=None,
        description='Nome ou trecho do nome da cooperativa para consultar feedbacks e avaliações.'
    )
    nota_minima: int = Field(default=1, description='Nota mínima (1 a 5) para filtrar avaliações.')
    limite: int = Field(default=5, description='Quantidade máxima de avaliações qualitativas retornadas.')

class RecomendacaoToolkit(Toolkit):
    nome = 'recomendacao_toolkit'
    descricao = 'Tools de recomendação multicritério de cooperativas no Neo4j.'
    repositories = {
        'recomendacao_repository': RecomendacaoRepository,
        'analitica_repository': AnaliticaRepository,
    }

    def __init__(self, repository: RecomendacaoRepository, repositories=None):
        self.repository = repository
        self.repositories = repositories or self.__class__.repositories

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

    def _resolver_empresa_id(
        self,
        config: Optional[RunnableConfig],
        empresa_id: Optional[str] = None,
    ) -> Optional[UUID]:
        if empresa_id:
            try:
                return UUID(str(empresa_id))
            except (ValueError, TypeError):
                pass
        cfg = ((config or {}).get('configurable') or {})
        raw_id = cfg.get('empresa_id')
        if raw_id:
            try:
                return UUID(str(raw_id))
            except (ValueError, TypeError):
                pass
        return None

    @Toolkit.timed(categoria='tool', nome='consultar_perfil_comprador_rfv')
    def consultar_perfil_comprador_rfv(
        self,
        config: Optional[RunnableConfig] = None,
        empresa_id: Optional[str] = None,
    ) -> dict:
        """Consulta o historico de compras e segmentacao RFV da empresa compradora no Renovai."""
        emp_uuid = self._resolver_empresa_id(config, empresa_id)
        if not emp_uuid:
            return {'status': 'error', 'message': 'Consulta exclusiva para empresas compradoras. empresa_id ausente.'}

        repo = self.repositories.get('analitica_repository')
        if not repo or not hasattr(repo, 'consultar_segmentacao_empresas_rfv'):
            return {'status': 'error', 'message': 'Repositorio analitico indisponivel.'}

        perfil = repo.consultar_segmentacao_empresas_rfv(empresa_id=emp_uuid)
        if not perfil:
            return {
                'status': 'ok',
                'empresa_id': str(emp_uuid),
                'mensagem': 'Empresa compradora ainda nao possui pedidos historicos faturados na plataforma.',
                'faixa_cliente': 'Novo Comprador',
            }

        return {
            'status': 'ok',
            'perfil_rfv': perfil,
        }

    @Toolkit.timed(categoria='tool', nome='consultar_avaliacoes_cooperativa')
    def consultar_avaliacoes_cooperativa(
        self,
        config: Optional[RunnableConfig] = None,
        cooperativa_nome: Optional[str] = None,
        nota_minima: int = 1,
        limite: int = 5,
    ) -> dict:
        """Consulta avaliacoes qualitativas e notas de compradores sobre a cooperativa parceira."""
        repo = self.repositories.get('analitica_repository')
        if not repo or not hasattr(repo, 'consultar_avaliacoes_cooperativa'):
            return {'status': 'error', 'message': 'Repositorio analitico indisponivel.'}

        avaliacoes = repo.consultar_avaliacoes_cooperativa(
            cooperativa_nome=cooperativa_nome,
            nota_minima=nota_minima,
            limite=limite,
        )
        return {
            'status': 'ok',
            'total_avaliacoes': len(avaliacoes),
            'avaliacoes': avaliacoes,
        }

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.recomendar_cooperativas,
                name='recomendar_cooperativas',
                description=self.recomendar_cooperativas.__doc__,
                args_schema=RecomendarCooperativasArgs,
            ),
            StructuredTool.from_function(
                func=self.consultar_perfil_comprador_rfv,
                name='consultar_perfil_comprador_rfv',
                description=self.consultar_perfil_comprador_rfv.__doc__,
                args_schema=ConsultarPerfilCompradorRfvArgs,
            ),
            StructuredTool.from_function(
                func=self.consultar_avaliacoes_cooperativa,
                name='consultar_avaliacoes_cooperativa',
                description=self.consultar_avaliacoes_cooperativa.__doc__,
                args_schema=ConsultarAvaliacoesCooperativaArgs,
            ),
        ]
