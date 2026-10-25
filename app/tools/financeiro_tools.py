from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID
import unicodedata

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.base import Toolkit
from app.repository.postgresql.rateio_repository import (
    RateioRepository,
    RateioSnapshot,
    Rateio,
)
from app.repository.postgresql.rateio_funcionario_repository import (
    RateioFuncionarioRepository,
    RateioFuncionarioSnapshot,
    RateioFuncionario,
)
from app.repository.postgresql.despesa_repository import (
    DespesaRepository,
    DespesaSnapshot,
    Despesa,
)
from app.repository.postgresql.funcionario_repository import (
    FuncionarioRepository,
    FuncionarioSnapshot,
    Funcionario,
)

_MESES_PT = {
    'janeiro': 1, 'jan': 1,
    'fevereiro': 2, 'fev': 2,
    'marco': 3, 'mar': 3,
    'abril': 4, 'abr': 4,
    'maio': 5, 'mai': 5,
    'junho': 6, 'jun': 6,
    'julho': 7, 'jul': 7,
    'agosto': 8, 'ago': 8,
    'setembro': 9, 'set': 9,
    'outubro': 10, 'out': 10,
    'novembro': 11, 'nov': 11,
    'dezembro': 12, 'dez': 12,
}


class ConsultarRateiosArgs(BaseModel):
    mes_referencia: Optional[str] = Field(
        default=None,
        description="Mês de referência do rateio (ex: '2026-09', '2026-09-01', 'Setembro/2026')."
    )
    tipo_rateio: Optional[str] = Field(
        default=None,
        description="Filtra pelo tipo de rateio (ex: 'AUTOMATICO', 'MANUAL', 'PROPORCIONAL')."
    )
    gestor_nome: Optional[str] = Field(
        default=None,
        description="Nome (ou parte do nome) do gestor responsável pelo fechamento do rateio."
    )


class ConsultarRateioIndividualArgs(BaseModel):
    cooperado_id: Optional[UUID] = Field(
        default=None,
        description="UUID do cooperado/funcionário para consulta do valor individual."
    )
    nome_cooperado: Optional[str] = Field(
        default=None,
        description="Nome ou parte do nome do cooperado para consulta de valor individual."
    )
    mes_referencia: Optional[str] = Field(
        default=None,
        description="Mês de referência para busca (ex: '2026-09', 'Setembro/2026')."
    )


class ConsultarDespesasArgs(BaseModel):
    mes_referencia: Optional[str] = Field(
        default=None,
        description="Mês de referência dos lançamentos de despesas operacionais (ex: '2026-09')."
    )
    nome_despesa: Optional[str] = Field(
        default=None,
        description="Nome ou categoria da despesa (ex: 'Energia', 'Água', 'Combustível', 'Manutenção')."
    )
    tipo_despesa: Optional[str] = Field(
        default=None,
        description="Tipo da despesa ('FIXA' ou 'VARIAVEL')."
    )
    apenas_ativas: bool = Field(
        default=False,
        description="Se true, filtra apenas despesas marcadas como ativas no cadastro."
    )


class ResumoFinanceiroMensalArgs(BaseModel):
    mes_referencia: Optional[str] = Field(
        default=None,
        description="Mês de referência para o fechamento/resumo financeiro (ex: '2026-09')."
    )


class FinanceiroToolkit(Toolkit):
    nome: str = 'financeiro_toolkit'
    descricao: str = 'Ferramentas para consulta de rateios entre cooperados, fechamentos mensais e despesas operacionais.'
    repositories = {
        'rateio_repository': RateioRepository,
        'rateio_funcionario_repository': RateioFuncionarioRepository,
        'despesa_repository': DespesaRepository,
        'funcionario_repository': FuncionarioRepository,
    }

    def __init__(self, repositories=None):
        self.repositories = repositories or self.__class__.repositories

    def _normalizar(self, texto: str) -> str:
        """Remove acentos, underline/hífen e baixa a caixa para comparação tolerante."""
        sem_acento = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
        return sem_acento.lower().replace('_', ' ').replace('-', ' ').strip()

    def _corresponde_nome(self, termo: str, alvo: str) -> bool:
        if not termo or not alvo:
            return False
        if termo in alvo or alvo in termo:
            return True
        if termo.endswith('s') and termo[:-1] in alvo:
            return True
        if alvo.endswith('s') and alvo[:-1] in termo:
            return True
        return False

    def _formatar_mes(self, d: date) -> str:
        return f'{d.year:04d}-{d.month:02d}'

    def _corresponde_mes(self, filtro: Optional[str], mes_ref: date) -> bool:
        """Verifica se o filtro textual de mês bate com a data mes_referencia."""
        if not filtro:
            return True
        f = self._normalizar(filtro)
        iso_ym = f'{mes_ref.year:04d} {mes_ref.month:02d}'
        iso_raw = f'{mes_ref.year:04d}-{mes_ref.month:02d}'
        br_ym = f'{mes_ref.month:02d}/{mes_ref.year:04d}'

        if filtro.strip().startswith(iso_raw) or f == iso_ym or filtro.strip() == br_ym:
            return True

        # Verifica nomes de mês em português (ex: "setembro/2026", "setembro 2026", "setembro")
        partes = f.replace('/', ' ').split()
        mes_num = None
        ano_num = None
        for p in partes:
            if p in _MESES_PT:
                mes_num = _MESES_PT[p]
            elif p.isdigit() and len(p) == 4:
                ano_num = int(p)
            elif p.isdigit() and 1 <= int(p) <= 12 and mes_num is None:
                mes_num = int(p)

        if mes_num is not None:
            if ano_num is not None:
                return mes_ref.year == ano_num and mes_ref.month == mes_num
            return mes_ref.month == mes_num

        return iso_raw in filtro.strip()

    @Toolkit.timed(categoria='tool', nome='consultar_rateios')
    def consultar_rateios(
        self,
        config: RunnableConfig,
        mes_referencia: Optional[str] = None,
        tipo_rateio: Optional[str] = None,
        gestor_nome: Optional[str] = None,
    ) -> dict:
        """Consulta rateios mensais realizados na cooperativa."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        rateio_snapshot: Optional[RateioSnapshot] = snapshots.get('rateio_snapshot')
        rateio_func_snapshot: Optional[RateioFuncionarioSnapshot] = snapshots.get('rateio_funcionario_snapshot')
        despesa_snapshot: Optional[DespesaSnapshot] = snapshots.get('despesa_snapshot')
        funcionario_snapshot: Optional[FuncionarioSnapshot] = snapshots.get('funcionario_snapshot')

        if not rateio_snapshot:
            return {'status': 'error', 'message': 'Snapshot de rateios indisponível para esta cooperativa.'}

        funcionarios_por_id: dict[UUID, Funcionario] = (
            {f.funcionario_id: f for f in funcionario_snapshot.itens}
            if funcionario_snapshot else {}
        )

        # Caso valor_total_rateio ou quantidade_cooperados não tenham vindo agregados na linha,
        # podemos complementar a partir de rateio_funcionario_snapshot.
        soma_por_rateio: dict[UUID, Decimal] = {}
        contagem_por_rateio: dict[UUID, int] = {}
        if rateio_func_snapshot:
            for rf in rateio_func_snapshot.itens:
                soma_por_rateio[rf.rateio_id] = soma_por_rateio.get(rf.rateio_id, Decimal(0)) + rf.valor_rateio
                contagem_por_rateio[rf.rateio_id] = contagem_por_rateio.get(rf.rateio_id, 0) + 1

        despesas_por_mes: dict[str, Decimal] = {}
        if despesa_snapshot:
            for d in despesa_snapshot.itens:
                chave = self._formatar_mes(d.mes_referencia)
                despesas_por_mes[chave] = despesas_por_mes.get(chave, Decimal(0)) + d.valor

        termo_tipo = self._normalizar(tipo_rateio) if tipo_rateio else None
        termo_gestor = self._normalizar(gestor_nome) if gestor_nome else None

        itens_filtrados: list[Rateio] = []
        for r in rateio_snapshot.itens:
            if not self._corresponde_mes(mes_referencia, r.mes_referencia):
                continue
            if termo_tipo and not self._corresponde_nome(termo_tipo, self._normalizar(r.tipo_rateio or '')):
                continue
            if termo_gestor:
                nome_g = r.nome_gestor or (
                    funcionarios_por_id[r.gestor_id].nome if r.gestor_id in funcionarios_por_id else ''
                )
                if not self._corresponde_nome(termo_gestor, self._normalizar(nome_g)):
                    continue
            itens_filtrados.append(r)

        rateios_detalhados = []
        for r in itens_filtrados:
            chave_mes = self._formatar_mes(r.mes_referencia)
            valor_total = r.valor_total_rateio or soma_por_rateio.get(r.rateio_id, Decimal(0))
            qtd_coop = r.quantidade_cooperados or contagem_por_rateio.get(r.rateio_id, 0)
            despesas_mes = despesas_por_mes.get(chave_mes, Decimal(0))
            nome_g = r.nome_gestor or (
                funcionarios_por_id[r.gestor_id].nome if r.gestor_id in funcionarios_por_id else None
            )

            rateios_detalhados.append({
                'rateio_id': str(r.rateio_id),
                'mes_referencia': chave_mes,
                'tipo_rateio': r.tipo_rateio,
                'descricao_tipo_rateio': r.descricao_tipo_rateio,
                'gestor_id': str(r.gestor_id),
                'nome_gestor': nome_g,
                'valor_liquido_distribuido': float(valor_total),
                'valor_despesas_operacionais': float(despesas_mes),
                'quantidade_cooperados': qtd_coop,
                'data_rateio': r.data_rateio.isoformat() if r.data_rateio else None,
            })

        return {
            'status': 'ok',
            'total_rateios_encontrados': len(rateios_detalhados),
            'rateios': rateios_detalhados,
        }

    @Toolkit.timed(categoria='tool', nome='consultar_rateio_individual')
    def consultar_rateio_individual(
        self,
        config: RunnableConfig,
        cooperado_id: Optional[UUID] = None,
        nome_cooperado: Optional[str] = None,
        mes_referencia: Optional[str] = None,
    ) -> dict:
        """Consulta valores individuais e percentuais distribuídos a cooperados em um rateio."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        rateio_func_snapshot: Optional[RateioFuncionarioSnapshot] = snapshots.get('rateio_funcionario_snapshot')
        rateio_snapshot: Optional[RateioSnapshot] = snapshots.get('rateio_snapshot')
        funcionario_snapshot: Optional[FuncionarioSnapshot] = snapshots.get('funcionario_snapshot')

        if not rateio_func_snapshot:
            return {'status': 'error', 'message': 'Snapshot de rateio por funcionário indisponível.'}

        funcionarios_por_id: dict[UUID, Funcionario] = (
            {f.funcionario_id: f for f in funcionario_snapshot.itens}
            if funcionario_snapshot else {}
        )

        totais_por_rateio: dict[UUID, Decimal] = {}
        if rateio_snapshot:
            for r in rateio_snapshot.itens:
                if r.valor_total_rateio > 0:
                    totais_por_rateio[r.rateio_id] = r.valor_total_rateio
        for rf in rateio_func_snapshot.itens:
            if rf.rateio_id not in totais_por_rateio:
                totais_por_rateio[rf.rateio_id] = Decimal(0)
        if not rateio_snapshot or any(v == 0 for v in totais_por_rateio.values()):
            soma_rf: dict[UUID, Decimal] = {}
            for rf in rateio_func_snapshot.itens:
                soma_rf[rf.rateio_id] = soma_rf.get(rf.rateio_id, Decimal(0)) + rf.valor_rateio
            for rid, soma in soma_rf.items():
                if totais_por_rateio.get(rid, Decimal(0)) == 0:
                    totais_por_rateio[rid] = soma

        termo_coop = self._normalizar(nome_cooperado) if nome_cooperado else None

        registros_filtrados: list[RateioFuncionario] = []
        for rf in rateio_func_snapshot.itens:
            if not self._corresponde_mes(mes_referencia, rf.mes_referencia):
                continue
            if cooperado_id and rf.cooperado_id != cooperado_id:
                continue
            if termo_coop:
                nome_c = rf.nome_cooperado or (
                    funcionarios_por_id[rf.cooperado_id].nome if rf.cooperado_id in funcionarios_por_id else ''
                )
                if not self._corresponde_nome(termo_coop, self._normalizar(nome_c)):
                    continue
            registros_filtrados.append(rf)

        cooperados_detalhados = []
        for rf in registros_filtrados:
            total_rateio = totais_por_rateio.get(rf.rateio_id, Decimal(0))
            pct = (
                round(float(rf.valor_rateio / total_rateio * 100), 2)
                if total_rateio > 0 else 0.0
            )
            func_info = funcionarios_por_id.get(rf.cooperado_id)
            nome_c = rf.nome_cooperado or (func_info.nome if func_info else None)

            cooperados_detalhados.append({
                'rateio_funcionario_id': str(rf.rateio_funcionario_id),
                'rateio_id': str(rf.rateio_id),
                'cooperado_id': str(rf.cooperado_id),
                'nome_cooperado': nome_c,
                'cargo': func_info.cargo if func_info else None,
                'mes_referencia': self._formatar_mes(rf.mes_referencia),
                'tipo_rateio': rf.tipo_rateio,
                'porcentagem_individual': pct,
                'valor_rateio': float(rf.valor_rateio),
                'valor_recebido': float(rf.valor_rateio),
                'data_rateio': rf.data_rateio.isoformat() if rf.data_rateio else None,
            })

        return {
            'status': 'ok',
            'total_registros': len(cooperados_detalhados),
            'cooperados': cooperados_detalhados,
        }

    @Toolkit.timed(categoria='tool', nome='consultar_despesas')
    def consultar_despesas(
        self,
        config: RunnableConfig,
        mes_referencia: Optional[str] = None,
        nome_despesa: Optional[str] = None,
        tipo_despesa: Optional[str] = None,
        apenas_ativas: bool = False,
        categoria: Optional[str] = None,
    ) -> dict:
        """Consulta lançamentos de despesas operacionais (fixas e variáveis) da cooperativa."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        despesa_snapshot: Optional[DespesaSnapshot] = snapshots.get('despesa_snapshot')

        if not despesa_snapshot:
            return {'status': 'error', 'message': 'Snapshot de despesas indisponível.'}

        termo_nome = self._normalizar(nome_despesa or categoria or '')
        termo_tipo = self._normalizar(tipo_despesa) if tipo_despesa else None

        despesas_filtradas: list[Despesa] = []
        for d in despesa_snapshot.itens:
            if apenas_ativas and not d.esta_ativa:
                continue
            if not self._corresponde_mes(mes_referencia, d.mes_referencia):
                continue
            if termo_tipo and self._normalizar(d.tipo_despesa) != termo_tipo:
                continue
            if termo_nome and not (
                self._corresponde_nome(termo_nome, self._normalizar(d.nome_despesa))
                or self._corresponde_nome(termo_nome, self._normalizar(d.tipo_despesa))
            ):
                continue
            despesas_filtradas.append(d)

        total_despesas = sum((d.valor or Decimal(0)) for d in despesas_filtradas)
        total_fixas = sum((d.valor or Decimal(0)) for d in despesas_filtradas if d.tipo_despesa.upper() == 'FIXA')
        total_variaveis = sum((d.valor or Decimal(0)) for d in despesas_filtradas if d.tipo_despesa.upper() == 'VARIAVEL')

        return {
            'status': 'ok',
            'total_despesas_encontradas': len(despesas_filtradas),
            'valor_total_despesas': float(total_despesas),
            'valor_despesas_fixas': float(total_fixas),
            'valor_despesas_variaveis': float(total_variaveis),
            'despesas': [
                {
                    'lancamento_id': str(d.lancamento_id),
                    'despesa_id': str(d.despesa_id),
                    'nome_despesa': d.nome_despesa,
                    'tipo_despesa': d.tipo_despesa,
                    'esta_ativa': d.esta_ativa,
                    'valor': float(d.valor),
                    'mes_referencia': self._formatar_mes(d.mes_referencia),
                    'data_lancamento': d.data_lancamento.isoformat() if d.data_lancamento else None,
                }
                for d in despesas_filtradas
            ],
        }

    @Toolkit.timed(categoria='tool', nome='resumo_financeiro_mensal')
    def resumo_financeiro_mensal(
        self,
        config: RunnableConfig,
        mes_referencia: Optional[str] = None,
    ) -> dict:
        """Gera um resumo consolidado de despesas operacionais (fixas/variáveis) e distribuição de rateio de um mês."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        rateio_snapshot: Optional[RateioSnapshot] = snapshots.get('rateio_snapshot')
        rateio_func_snapshot: Optional[RateioFuncionarioSnapshot] = snapshots.get('rateio_funcionario_snapshot')
        despesa_snapshot: Optional[DespesaSnapshot] = snapshots.get('despesa_snapshot')
        funcionario_snapshot: Optional[FuncionarioSnapshot] = snapshots.get('funcionario_snapshot')

        if not rateio_snapshot and not despesa_snapshot:
            return {'status': 'error', 'message': 'Snapshots financeiros de rateios e despesas indisponíveis.'}

        rateios = list(rateio_snapshot.itens) if rateio_snapshot else []
        despesas = list(despesa_snapshot.itens) if despesa_snapshot else []

        if mes_referencia:
            rateios = [r for r in rateios if self._corresponde_mes(mes_referencia, r.mes_referencia)]
            despesas = [d for d in despesas if self._corresponde_mes(mes_referencia, d.mes_referencia)]
        else:
            # Seleciona o mês mais recente disponível entre rateios e despesas
            meses_disponiveis = sorted(
                {r.mes_referencia for r in rateios} | {d.mes_referencia for d in despesas},
                reverse=True,
            )
            if meses_disponiveis:
                mes_alvo = meses_disponiveis[0]
                rateios = [r for r in rateios if r.mes_referencia == mes_alvo]
                despesas = [d for d in despesas if d.mes_referencia == mes_alvo]

        if not rateios and not despesas:
            return {
                'status': 'ok',
                'message': f'Nenhum registro financeiro encontrado para o período {mes_referencia or "solicitado"}.',
            }

        mes_ref_date = rateios[0].mes_referencia if rateios else despesas[0].mes_referencia
        chave_mes = self._formatar_mes(mes_ref_date)

        despesas_fixas = sum((d.valor or Decimal(0)) for d in despesas if d.tipo_despesa.upper() == 'FIXA')
        despesas_variaveis = sum((d.valor or Decimal(0)) for d in despesas if d.tipo_despesa.upper() == 'VARIAVEL')
        total_despesas = sum((d.valor or Decimal(0)) for d in despesas)

        r = rateios[0] if rateios else None
        valor_liquido_rateado = Decimal(0)
        cooperados_beneficiados = 0
        if r:
            valor_liquido_rateado = r.valor_total_rateio
            cooperados_beneficiados = r.quantidade_cooperados
            if (valor_liquido_rateado == 0 or cooperados_beneficiados == 0) and rateio_func_snapshot:
                rfs_do_rateio = [rf for rf in rateio_func_snapshot.itens if rf.rateio_id == r.rateio_id]
                if valor_liquido_rateado == 0:
                    valor_liquido_rateado = sum((rf.valor_rateio for rf in rfs_do_rateio), Decimal(0))
                if cooperados_beneficiados == 0:
                    cooperados_beneficiados = len(rfs_do_rateio)

        cooperados_ativos = (
            sum(1 for f in funcionario_snapshot.itens if f.esta_ativo)
            if funcionario_snapshot else cooperados_beneficiados
        )

        return {
            'status': 'ok',
            'mes_referencia': chave_mes,
            'rateio_realizado': r is not None,
            'rateio_id': str(r.rateio_id) if r else None,
            'tipo_rateio': r.tipo_rateio if r else None,
            'nome_gestor': r.nome_gestor if r else None,
            'despesas_fixas': float(despesas_fixas),
            'despesas_variaveis': float(despesas_variaveis),
            'despesas_operacionais': float(total_despesas),
            'valor_liquido_distribuido': float(valor_liquido_rateado),
            'cooperados_beneficiados': cooperados_beneficiados,
            'cooperados_ativos': cooperados_ativos,
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
