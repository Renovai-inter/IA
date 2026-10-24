from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.base import Toolkit
from app.repository.postgresql.coleta_repository import ColetaSnapshot, Coleta, Triagem


class ConsultarColetasArgs(BaseModel):
    status: Optional[str] = Field(
        default=None,
        description='Filtra por status da coleta (ex: \'AGENDADA\', \'PENDENTE\', \'CONCLUIDA\', \'CANCELADA\').'
    )
    tipo_coleta: Optional[str] = Field(
        default=None,
        description='Filtra por tipo de coleta (ex: \'EXTERNA\', \'INTERNA\', \'IMEDIATA\').'
    )
    equipe: Optional[str] = Field(
        default=None,
        description='Nome ou identificador da equipe responsável.'
    )
    limite: int = Field(default=20, description='Número máximo de coletas retornadas.')


class ConsultarTriagensArgs(BaseModel):
    status: Optional[str] = Field(
        default=None,
        description='Filtra por status da triagem (ex: \'EM_ANDAMENTO\', \'CONCLUIDA\').'
    )
    equipe: Optional[str] = Field(
        default=None,
        description='Nome ou identificador da equipe de triagem.'
    )
    limite: int = Field(default=20, description='Número máximo de triagens retornadas.')


class ResumoOperacionalArgs(BaseModel):
    pass


class ColetaToolkit(Toolkit):
    nome: str = 'coleta_toolkit'
    descricao: str = 'Ferramentas para consulta de coletas e triagens operacionais de cooperativas.'

    def __init__(self, repositories=None):
        self.repositories = repositories or {}

    @Toolkit.timed(categoria='tool', nome='consultar_coletas')
    def consultar_coletas(
        self,
        config: RunnableConfig,
        status: Optional[str] = None,
        tipo_coleta: Optional[str] = None,
        equipe: Optional[str] = None,
        limite: int = 20,
    ) -> dict:
        """Consulta registros de coletas realizadas ou agendadas da cooperativa."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        coleta_snapshot: ColetaSnapshot | None = snapshots.get('coleta_snapshot')

        if not coleta_snapshot:
            return {'status': 'error', 'message': 'Snapshot de coletas indisponível para esta cooperativa.'}

        itens = coleta_snapshot.itens
        if status:
            itens = [c for c in itens if c.status.upper() == status.upper()]
        if tipo_coleta:
            itens = [c for c in itens if c.tipo_coleta and c.tipo_coleta.upper() == tipo_coleta.upper()]
        if equipe:
            itens = [c for c in itens if c.equipe_responsavel and equipe.lower() in c.equipe_responsavel.lower()]

        itens_retornados = itens[:limite]
        peso_total_real = sum((c.peso_real_kg or Decimal(0)) for c in itens)
        peso_total_estimado = sum((c.peso_estimado_kg or Decimal(0)) for c in itens)

        return {
            'status': 'ok',
            'total_coletas_encontradas': len(itens),
            'peso_total_real_kg': float(peso_total_real),
            'peso_total_estimado_kg': float(peso_total_estimado),
            'coletas': [
                {
                    'coleta_id': str(c.coleta_id),
                    'codigo': c.codigo,
                    'data_coleta': c.data_coleta.isoformat() if c.data_coleta else None,
                    'status': c.status,
                    'tipo_coleta': c.tipo_coleta,
                    'peso_real_kg': float(c.peso_real_kg or 0),
                    'peso_estimado_kg': float(c.peso_estimado_kg or 0),
                    'ponto_coleta': c.ponto_coleta,
                    'equipe_responsavel': c.equipe_responsavel,
                }
                for c in itens_retornados
            ],
        }

    @Toolkit.timed(categoria='tool', nome='consultar_triagens')
    def consultar_triagens(
        self,
        config: RunnableConfig,
        status: Optional[str] = None,
        equipe: Optional[str] = None,
        limite: int = 20,
    ) -> dict:
        """Consulta registros de triagens e separação de materiais na cooperativa."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        coleta_snapshot: ColetaSnapshot | None = snapshots.get('coleta_snapshot')

        if not coleta_snapshot:
            return {'status': 'error', 'message': 'Snapshot de triagens indisponível para esta cooperativa.'}

        triagens = coleta_snapshot.triagens
        if status:
            triagens = [t for t in triagens if t.status.upper() == status.upper()]
        if equipe:
            triagens = [t for t in triagens if t.equipe and equipe.lower() in t.equipe.lower()]

        itens_retornados = triagens[:limite]
        peso_total_triado = sum(t.peso_total_kg for t in triagens)
        peso_total_rejeito = sum((t.peso_rejeito_kg or Decimal(0)) for t in triagens)
        taxa_rejeito = float((peso_total_rejeito / peso_total_triado * 100)) if peso_total_triado > 0 else 0.0

        return {
            'status': 'ok',
            'total_triagens_encontradas': len(triagens),
            'peso_total_triado_kg': float(peso_total_triado),
            'peso_total_rejeito_kg': float(peso_total_rejeito),
            'taxa_rejeito_percentual': round(taxa_rejeito, 2),
            'triagens': [
                {
                    'triagem_id': str(t.triagem_id),
                    'coleta_id': str(t.coleta_id) if t.coleta_id else None,
                    'data_triagem': t.data_triagem.isoformat() if t.data_triagem else None,
                    'status': t.status,
                    'peso_total_kg': float(t.peso_total_kg),
                    'peso_rejeito_kg': float(t.peso_rejeito_kg or 0),
                    'equipe': t.equipe,
                }
                for t in itens_retornados
            ],
        }

    @Toolkit.timed(categoria='tool', nome='resumo_operacional')
    def resumo_operacional(self, config: RunnableConfig) -> dict:
        """Gera um resumo integrado das operações recentes (coletas e triagens)."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        coleta_snapshot: ColetaSnapshot | None = snapshots.get('coleta_snapshot')

        if not coleta_snapshot:
            return {'status': 'error', 'message': 'Snapshot de operações indisponível.'}

        total_coletas = len(coleta_snapshot.itens)
        total_triagens = len(coleta_snapshot.triagens)
        peso_coletas = sum((c.peso_real_kg or Decimal(0)) for c in coleta_snapshot.itens)
        peso_triagens = sum(t.peso_total_kg for t in coleta_snapshot.triagens)
        peso_rejeito = sum((t.peso_rejeito_kg or Decimal(0)) for t in coleta_snapshot.triagens)

        return {
            'status': 'ok',
            'total_coletas': total_coletas,
            'peso_coletado_kg': float(peso_coletas),
            'total_triagens': total_triagens,
            'peso_triado_kg': float(peso_triagens),
            'peso_rejeito_kg': float(peso_rejeito),
            'taxa_aproveitamento_percentual': round(float((peso_triagens - peso_rejeito) / peso_triagens * 100), 2) if peso_triagens > 0 else 100.0,
        }

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.consultar_coletas,
                name='consultar_coletas',
                description=self.consultar_coletas.__doc__,
                args_schema=ConsultarColetasArgs,
            ),
            StructuredTool.from_function(
                func=self.consultar_triagens,
                name='consultar_triagens',
                description=self.consultar_triagens.__doc__,
                args_schema=ConsultarTriagensArgs,
            ),
            StructuredTool.from_function(
                func=self.resumo_operacional,
                name='resumo_operacional',
                description=self.resumo_operacional.__doc__,
                args_schema=ResumoOperacionalArgs,
            ),
        ]
