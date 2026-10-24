"""Testes unitários das tools de Coleta & Triagem."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.repository.postgresql.coleta_repository import ColetaSnapshot, Coleta, Triagem
from app.tools.coleta_tools import ColetaToolkit


@pytest.fixture
def snapshot_coleta_teste():
    coop_id = uuid4()
    c1 = Coleta(
        coleta_id=uuid4(),
        cooperativa_id=coop_id,
        codigo='COL-001',
        data_coleta=datetime(2026, 10, 24, 10, 0),
        status='CONCLUIDA',
        tipo_coleta='EXTERNA',
        peso_estimado_kg=Decimal('500'),
        peso_real_kg=Decimal('520'),
        ponto_coleta='Bairro Centro',
        equipe_responsavel='Equipe Alfa',
    )
    c2 = Coleta(
        coleta_id=uuid4(),
        cooperativa_id=coop_id,
        codigo='COL-002',
        data_coleta=datetime(2026, 10, 24, 14, 0),
        status='AGENDADA',
        tipo_coleta='INTERNA',
        peso_estimado_kg=Decimal('300'),
        peso_real_kg=Decimal('0'),
        ponto_coleta='Galpão 2',
        equipe_responsavel='Equipe Beta',
    )
    t1 = Triagem(
        triagem_id=uuid4(),
        cooperativa_id=coop_id,
        coleta_id=c1.coleta_id,
        data_triagem=datetime(2026, 10, 24, 11, 0),
        status='CONCLUIDA',
        peso_total_kg=Decimal('520'),
        peso_rejeito_kg=Decimal('20'),
        equipe='Equipe Triagem 1',
    )
    return ColetaSnapshot(
        cooperativa_id=coop_id,
        capturado_em=datetime.now(),
        itens=[c1, c2],
        triagens=[t1],
    )


def test_consultar_coletas_filtro_status(snapshot_coleta_teste):
    toolkit = ColetaToolkit()
    config = {'configurable': {'snapshots': {'coleta_snapshot': snapshot_coleta_teste}}}

    res = toolkit.consultar_coletas(config=config, status='CONCLUIDA')
    assert res['status'] == 'ok'
    assert res['total_coletas_encontradas'] == 1
    assert res['coletas'][0]['codigo'] == 'COL-001'
    assert res['peso_total_real_kg'] == 520.0


def test_consultar_triagens_e_taxa_rejeito(snapshot_coleta_teste):
    toolkit = ColetaToolkit()
    config = {'configurable': {'snapshots': {'coleta_snapshot': snapshot_coleta_teste}}}

    res = toolkit.consultar_triagens(config=config)
    assert res['status'] == 'ok'
    assert res['total_triagens_encontradas'] == 1
    assert res['peso_total_triado_kg'] == 520.0
    assert res['peso_total_rejeito_kg'] == 20.0
    assert res['taxa_rejeito_percentual'] == round(20 / 520 * 100, 2)


def test_resumo_operacional(snapshot_coleta_teste):
    toolkit = ColetaToolkit()
    config = {'configurable': {'snapshots': {'coleta_snapshot': snapshot_coleta_teste}}}

    res = toolkit.resumo_operacional(config=config)
    assert res['status'] == 'ok'
    assert res['total_coletas'] == 2
    assert res['total_triagens'] == 1
    assert res['taxa_aproveitamento_percentual'] > 90.0
