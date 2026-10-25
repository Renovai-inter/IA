"""Testes unitários das tools de Financeiro, Rateios e Despesas."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.repository.postgresql.rateio_repository import RateioSnapshot, Rateio
from app.repository.postgresql.rateio_funcionario_repository import (
    RateioFuncionarioSnapshot,
    RateioFuncionario,
)
from app.repository.postgresql.despesa_repository import DespesaSnapshot, Despesa
from app.repository.postgresql.funcionario_repository import FuncionarioSnapshot, Funcionario
from app.tools.financeiro_tools import FinanceiroToolkit


@pytest.fixture
def snapshots_financeiro_teste():
    coop_id = uuid4()
    gestor_id = uuid4()
    coop_maria_id = uuid4()
    rateio_id = uuid4()
    agora = datetime(2026, 9, 30, 18, 0)

    f_maria = Funcionario(
        funcionario_id=coop_maria_id,
        usuario_id=uuid4(),
        cooperativa_id=coop_id,
        nome='Maria Santos',
        email='maria@coop.com',
        cpf='123.456.789-00',
        cargo_id=uuid4(),
        cargo='Cooperado',
        status_funcionario='ATIVO',
        esta_ativo=True,
        esta_afastado=False,
        data_admissao=datetime(2025, 1, 10, 8, 0),
    )

    r1 = Rateio(
        rateio_id=rateio_id,
        cooperativa_id=coop_id,
        gestor_id=gestor_id,
        nome_gestor='Carlos Gestor',
        tipo_rateio_id=uuid4(),
        tipo_rateio='AUTOMATICO',
        descricao_tipo_rateio='Rateio igualitário entre cooperados ativos',
        mes_referencia=date(2026, 9, 1),
        data_rateio=agora,
        valor_total_rateio=Decimal('25000.00'),
        quantidade_cooperados=10,
    )

    rf1 = RateioFuncionario(
        rateio_funcionario_id=uuid4(),
        rateio_id=rateio_id,
        cooperativa_id=coop_id,
        cooperado_id=coop_maria_id,
        nome_cooperado='Maria Santos',
        mes_referencia=date(2026, 9, 1),
        tipo_rateio='AUTOMATICO',
        valor_rateio=Decimal('2500.00'),
        data_rateio=agora,
    )

    d1 = Despesa(
        lancamento_id=uuid4(),
        despesa_id=uuid4(),
        cooperativa_id=coop_id,
        nome_despesa='Energia Elétrica Galpão',
        tipo_despesa='FIXA',
        esta_ativa=True,
        valor=Decimal('1800.00'),
        mes_referencia=date(2026, 9, 1),
        data_lancamento=datetime(2026, 9, 15, 10, 0),
    )
    d2 = Despesa(
        lancamento_id=uuid4(),
        despesa_id=uuid4(),
        cooperativa_id=coop_id,
        nome_despesa='Combustível Caminhão',
        tipo_despesa='VARIAVEL',
        esta_ativa=True,
        valor=Decimal('3200.00'),
        mes_referencia=date(2026, 9, 1),
        data_lancamento=datetime(2026, 9, 20, 14, 0),
    )

    return {
        'rateio_snapshot': RateioSnapshot(cooperativa_id=coop_id, capturado_em=agora, itens=[r1]),
        'rateio_funcionario_snapshot': RateioFuncionarioSnapshot(
            cooperativa_id=coop_id, capturado_em=agora, itens=[rf1]
        ),
        'despesa_snapshot': DespesaSnapshot(cooperativa_id=coop_id, capturado_em=agora, itens=[d1, d2]),
        'funcionario_snapshot': FuncionarioSnapshot(cooperativa_id=coop_id, capturado_em=agora, itens=[f_maria]),
    }


def test_consultar_rateios(snapshots_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': snapshots_financeiro_teste}}

    res = toolkit.consultar_rateios(config=config, mes_referencia='2026-09')
    assert res['status'] == 'ok'
    assert res['total_rateios_encontrados'] == 1
    assert res['rateios'][0]['valor_liquido_distribuido'] == 25000.0
    assert res['rateios'][0]['valor_despesas_operacionais'] == 5000.0


def test_consultar_rateio_individual(snapshots_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': snapshots_financeiro_teste}}

    res = toolkit.consultar_rateio_individual(config=config, nome_cooperado='Maria')
    assert res['status'] == 'ok'
    assert res['total_registros'] == 1
    assert res['cooperados'][0]['valor_recebido'] == 2500.0
    assert res['cooperados'][0]['porcentagem_individual'] == 10.0


def test_consultar_despesas(snapshots_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': snapshots_financeiro_teste}}

    res = toolkit.consultar_despesas(config=config, nome_despesa='Energia')
    assert res['status'] == 'ok'
    assert res['total_despesas_encontradas'] == 1
    assert res['valor_total_despesas'] == 1800.0


def test_resumo_financeiro_mensal(snapshots_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': snapshots_financeiro_teste}}

    res = toolkit.resumo_financeiro_mensal(config=config, mes_referencia='Setembro/2026')
    assert res['status'] == 'ok'
    assert res['mes_referencia'] == '2026-09'
    assert res['despesas_fixas'] == 1800.0
    assert res['despesas_variaveis'] == 3200.0
    assert res['despesas_operacionais'] == 5000.0
    assert res['valor_liquido_distribuido'] == 25000.0
