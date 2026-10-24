"""Testes unitários das tools de Financeiro & Rateio."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.repository.postgresql.financeiro_repository import (
    FinanceiroSnapshot,
    Rateio,
    RateioFuncionario,
    Despesa,
)
from app.tools.financeiro_tools import FinanceiroToolkit


@pytest.fixture
def snapshot_financeiro_teste():
    coop_id = uuid4()
    r1 = Rateio(
        rateio_id=uuid4(),
        cooperativa_id=coop_id,
        mes_referencia='2026-09',
        tipo_rateio='PROPORCIONAL_HORAS',
        status='FECHADO',
        valor_total_arrecadado=Decimal('30000.00'),
        valor_despesas_operacionais=Decimal('5000.00'),
        valor_liquido_distribuido=Decimal('25000.00'),
        quantidade_cooperados=10,
        data_abertura=datetime(2026, 9, 1),
        data_fechamento=datetime(2026, 9, 30),
    )
    rf1 = RateioFuncionario(
        rateio_funcionario_id=uuid4(),
        rateio_id=r1.rateio_id,
        cooperativa_id=coop_id,
        cooperado_id=uuid4(),
        nome_cooperado='Maria Santos',
        mes_referencia='2026-09',
        porcentagem_individual=Decimal('10.00'),
        valor_recebido=Decimal('2500.00'),
        horas_trabalhadas=Decimal('160'),
        peso_triado_kg=None,
    )
    d1 = Despesa(
        despesa_id=uuid4(),
        cooperativa_id=coop_id,
        categoria='ENERGIA',
        descricao='Conta de luz galpão',
        valor=Decimal('1800.00'),
        data_despesa=datetime(2026, 9, 15),
        mes_referencia='2026-09',
        status_pagamento='PAGO',
    )
    return FinanceiroSnapshot(
        cooperativa_id=coop_id,
        capturado_em=datetime.now(),
        itens=[r1],
        itens_funcionarios=[rf1],
        despesas=[d1],
    )


def test_consultar_rateios(snapshot_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': {'financeiro_snapshot': snapshot_financeiro_teste}}}

    res = toolkit.consultar_rateios(config=config, mes_referencia='2026-09')
    assert res['status'] == 'ok'
    assert res['total_rateios_encontrados'] == 1
    assert res['rateios'][0]['valor_liquido_distribuido'] == 25000.0


def test_consultar_rateio_individual(snapshot_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': {'financeiro_snapshot': snapshot_financeiro_teste}}}

    res = toolkit.consultar_rateio_individual(config=config, nome_cooperado='Maria')
    assert res['status'] == 'ok'
    assert res['total_registros'] == 1
    assert res['cooperados'][0]['valor_recebido'] == 2500.0


def test_consultar_despesas(snapshot_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': {'financeiro_snapshot': snapshot_financeiro_teste}}}

    res = toolkit.consultar_despesas(config=config, categoria='ENERGIA')
    assert res['status'] == 'ok'
    assert res['total_despesas_encontradas'] == 1
    assert res['valor_total_despesas'] == 1800.0


def test_resumo_financeiro_mensal(snapshot_financeiro_teste):
    toolkit = FinanceiroToolkit()
    config = {'configurable': {'snapshots': {'financeiro_snapshot': snapshot_financeiro_teste}}}

    res = toolkit.resumo_financeiro_mensal(config=config, mes_referencia='2026-09')
    assert res['status'] == 'ok'
    assert res['valor_arrecadado'] == 30000.0
    assert res['valor_liquido_distribuido'] == 25000.0
