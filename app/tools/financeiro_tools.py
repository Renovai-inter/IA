from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.base import Toolkit
from app.repository.postgresql.financeiro_repository import (
    FinanceiroSnapshot,
    Rateio,
    RateioFuncionario,
    Despesa,
)


class ConsultarRateiosArgs(BaseModel):
    mes_referencia: Optional[str] = Field(
        default=None,
        description='Mês de referência do rateio (ex: \'2026-09\', \'Setembro/2026\').'
    )
    status: Optional[str] = Field(
        default=None,
        description='Status do rateio (ex: \'ABERTO\', \'FECHADO\', \'EM_PROCESSAMENTO\').'
    )


class ConsultarRateioIndividualArgs(BaseModel):
    nome_cooperado: Optional[str] = Field(
        default=None,
        description='Nome ou parte do nome do cooperado para consulta de valor individual.'
    )
    mes_referencia: Optional[str] = Field(
        default=None,
        description='Mês de referência para busca.'
    )


class ConsultarDespesasArgs(BaseModel):
    mes_referencia: Optional[str] = Field(
        default=None,
        description='Mês de referência das despesas operacionais.'
    )
    categoria: Optional[str] = Field(
        default=None,
        description='Categoria da despesa (ex: \'ENERGIA\', \'AGUA\', \'COMBUSTIVEL\', \'MANUTENCAO\').'
    )


class ResumoFinanceiroMensalArgs(BaseModel):
    mes_referencia: Optional[str] = Field(
        default=None,
        description='Mês de referência para o fechamento/resumo financeiro.'
    )


class FinanceiroToolkit(Toolkit):
    nome: str = 'financeiro_toolkit'
    descricao: str = 'Ferramentas para consulta de rateios entre cooperados, fechamentos e despesas.'

    def __init__(self, repositories=None):
        self.repositories = repositories or {}

    @Toolkit.timed(categoria='tool', nome='consultar_rateios')
    def consultar_rateios(
        self,
        config: RunnableConfig,
        mes_referencia: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """Consulta rateios gerais abertos ou fechados na cooperativa."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        fin_snapshot: FinanceiroSnapshot | None = snapshots.get('financeiro_snapshot')

        if not fin_snapshot:
            return {'status': 'error', 'message': 'Snapshot financeiro indisponível para esta cooperativa.'}

        itens = fin_snapshot.itens
        if mes_referencia:
            itens = [r for r in itens if mes_referencia.lower() in r.mes_referencia.lower()]
        if status:
            itens = [r for r in itens if r.status.upper() == status.upper()]

        return {
            'status': 'ok',
            'total_rateios_encontrados': len(itens),
            'rateios': [
                {
                    'rateio_id': str(r.rateio_id),
                    'mes_referencia': r.mes_referencia,
                    'tipo_rateio': r.tipo_rateio,
                    'status': r.status,
                    'valor_total_arrecadado': float(r.valor_total_arrecadado),
                    'valor_despesas_operacionais': float(r.valor_despesas_operacionais),
                    'valor_liquido_distribuido': float(r.valor_liquido_distribuido),
                    'quantidade_cooperados': r.quantidade_cooperados,
                    'data_abertura': r.data_abertura.isoformat() if r.data_abertura else None,
                    'data_fechamento': r.data_fechamento.isoformat() if r.data_fechamento else None,
                }
                for r in itens
            ],
        }

    @Toolkit.timed(categoria='tool', nome='consultar_rateio_individual')
    def consultar_rateio_individual(
        self,
        config: RunnableConfig,
        nome_cooperado: Optional[str] = None,
        mes_referencia: Optional[str] = None,
    ) -> dict:
        """Consulta valores individuais e percentuais distribuídos a cooperados em um rateio."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        fin_snapshot: FinanceiroSnapshot | None = snapshots.get('financeiro_snapshot')

        if not fin_snapshot:
            return {'status': 'error', 'message': 'Snapshot financeiro indisponível.'}

        funcionarios = fin_snapshot.itens_funcionarios
        if mes_referencia:
            funcionarios = [f for f in funcionarios if mes_referencia.lower() in f.mes_referencia.lower()]
        if nome_cooperado:
            funcionarios = [f for f in funcionarios if nome_cooperado.lower() in f.nome_cooperado.lower()]

        return {
            'status': 'ok',
            'total_registros': len(funcionarios),
            'cooperados': [
                {
                    'nome_cooperado': f.nome_cooperado,
                    'mes_referencia': f.mes_referencia,
                    'porcentagem_individual': float(f.porcentagem_individual),
                    'valor_recebido': float(f.valor_recebido),
                    'horas_trabalhadas': float(f.horas_trabalhadas) if f.horas_trabalhadas else None,
                    'peso_triado_kg': float(f.peso_triado_kg) if f.peso_triado_kg else None,
                }
                for f in funcionarios
            ],
        }

    @Toolkit.timed(categoria='tool', nome='consultar_despesas')
    def consultar_despesas(
        self,
        config: RunnableConfig,
        mes_referencia: Optional[str] = None,
        categoria: Optional[str] = None,
    ) -> dict:
        """Consulta despesas operacionais da cooperativa deduzidas no rateio."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        fin_snapshot: FinanceiroSnapshot | None = snapshots.get('financeiro_snapshot')

        if not fin_snapshot:
            return {'status': 'error', 'message': 'Snapshot financeiro indisponível.'}

        despesas = fin_snapshot.despesas
        if mes_referencia:
            despesas = [d for d in despesas if mes_referencia.lower() in d.mes_referencia.lower()]
        if categoria:
            despesas = [d for d in despesas if categoria.lower() in d.categoria.lower()]

        total_despesas = sum(d.valor for d in despesas)

        return {
            'status': 'ok',
            'total_despesas_encontradas': len(despesas),
            'valor_total_despesas': float(total_despesas),
            'despesas': [
                {
                    'despesa_id': str(d.despesa_id),
                    'categoria': d.categoria,
                    'descricao': d.descricao,
                    'valor': float(d.valor),
                    'mes_referencia': d.mes_referencia,
                    'status_pagamento': d.status_pagamento,
                }
                for d in despesas
            ],
        }

    @Toolkit.timed(categoria='tool', nome='resumo_financeiro_mensal')
    def resumo_financeiro_mensal(
        self,
        config: RunnableConfig,
        mes_referencia: Optional[str] = None,
    ) -> dict:
        """Gera um resumo consolidado de arrecadação, despesas e distribuição líquida de um mês."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        fin_snapshot: FinanceiroSnapshot | None = snapshots.get('financeiro_snapshot')

        if not fin_snapshot:
            return {'status': 'error', 'message': 'Snapshot financeiro indisponível.'}

        rateios = fin_snapshot.itens
        if mes_referencia:
            rateios = [r for r in rateios if mes_referencia.lower() in r.mes_referencia.lower()]

        if not rateios:
            return {'status': 'ok', 'message': f'Nenhum rateio registrado para o período {mes_referencia or "solicitado"}.'}

        r = rateios[0]
        return {
            'status': 'ok',
            'mes_referencia': r.mes_referencia,
            'status_rateio': r.status,
            'tipo_rateio': r.tipo_rateio,
            'valor_arrecadado': float(r.valor_total_arrecadado),
            'despesas_operacionais': float(r.valor_despesas_operacionais),
            'valor_liquido_distribuido': float(r.valor_liquido_distribuido),
            'cooperados_beneficiados': r.quantidade_cooperados,
        }

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.consultar_rateios,
                name='consultar_rateios',
                description=self.consultar_rateios.__doc__,
                args_schema=ConsultarRateiosArgs,
            ),
            StructuredTool.from_function(
                func=self.consultar_rateio_individual,
                name='consultar_rateio_individual',
                description=self.consultar_rateio_individual.__doc__,
                args_schema=ConsultarRateioIndividualArgs,
            ),
            StructuredTool.from_function(
                func=self.consultar_despesas,
                name='consultar_despesas',
                description=self.consultar_despesas.__doc__,
                args_schema=ConsultarDespesasArgs,
            ),
            StructuredTool.from_function(
                func=self.resumo_financeiro_mensal,
                name='resumo_financeiro_mensal',
                description=self.resumo_financeiro_mensal.__doc__,
                args_schema=ResumoFinanceiroMensalArgs,
            ),
        ]
