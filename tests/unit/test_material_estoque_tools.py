from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from app.repository.postgresql.estoque_repository import Estoque, EstoqueSnapshot
from app.repository.postgresql.material_repository import Material, MaterialSnapshot
from app.repository.postgresql.movimentacao_estoque_repository import (
    MovimentacaoEstoque,
    MovimentacaoEstoqueSnapshot,
)
from app.tools.material_estoque_tools import MaterialEstoqueToolkit


@pytest.fixture
def toolkit():
    return MaterialEstoqueToolkit()


@pytest.fixture
def mock_snapshots():
    cooperativa_id = UUID("dc2e31e4-147d-4737-9c48-f7a9cb298bdd")
    cat_metal_id = UUID("11111111-1111-1111-1111-111111111111")
    cat_plastico_id = UUID("22222222-2222-2222-2222-222222222222")

    mat_latinha_id = UUID("d98fdab6-94bf-4b6c-b968-792093422037")
    mat_pet_id = UUID("33333333-3333-3333-3333-333333333333")

    # Simula cenário onde material_snapshot tem APENAS Latinha (ex: cooperativa customizou preço só de Latinha)
    material_snapshot = MaterialSnapshot(
        cooperativa_id=cooperativa_id,
        capturado_em=datetime.now(timezone.utc),
        itens=[
            Material(
                material_id=mat_latinha_id,
                categoria_id=UUID("44444444-4444-4444-4444-444444444444"),
                nome_categoria="Latinha",
                categoria_pai_id=cat_metal_id,
                nome_categoria_pai="Metal",
                cooperativa_id=cooperativa_id,
                nome_cooperativa="Cooperativa Central",
                preco_sugerido=Decimal("6.09"),
                esta_disponivel=True,
            )
        ],
    )

    # Estoque tem Latinha e PET
    estoque_snapshot = EstoqueSnapshot(
        cooperativa_id=cooperativa_id,
        capturado_em=datetime.now(timezone.utc),
        itens=[
            Estoque(
                estoque_id=UUID("55555555-5555-5555-5555-555555555555"),
                cooperativa_id=cooperativa_id,
                nome_cooperativa="Cooperativa Central",
                material_id=mat_latinha_id,
                nome_categoria="Latinha",
                categoria_pai_id=cat_metal_id,
                nome_categoria_pai="Metal",
                quantidade_kg=Decimal("150.0"),
                data_atualizacao=datetime.now(timezone.utc),
            ),
            Estoque(
                estoque_id=UUID("66666666-6666-6666-6666-666666666666"),
                cooperativa_id=cooperativa_id,
                nome_cooperativa="Cooperativa Central",
                material_id=mat_pet_id,
                nome_categoria="Garrafa PET",
                categoria_pai_id=cat_plastico_id,
                nome_categoria_pai="Plástico",
                quantidade_kg=Decimal("300.0"),
                data_atualizacao=datetime.now(timezone.utc),
            ),
        ],
    )

    # Movimentações têm Latinha e PET
    movimentacao_snapshot = MovimentacaoEstoqueSnapshot(
        cooperativa_id=cooperativa_id,
        capturado_em=datetime.now(timezone.utc),
        itens=[
            MovimentacaoEstoque(
                movimentacao_id=UUID("77777777-7777-7777-7777-777777777777"),
                estoque_id=UUID("55555555-5555-5555-5555-555555555555"),
                cooperativa_id=cooperativa_id,
                material_id=mat_latinha_id,
                nome_categoria="Latinha",
                categoria_pai_id=cat_metal_id,
                nome_categoria_pai="Metal",
                triagem_id=None,
                item_id=None,
                quantidade_kg=Decimal("150.0"),
                tipo_movimentacao="ENTRADA",
                data_movimentacao=datetime(2026, 9, 21, 12, 0, 0),
            ),
            MovimentacaoEstoque(
                movimentacao_id=UUID("88888888-8888-8888-8888-888888888888"),
                estoque_id=UUID("66666666-6666-6666-6666-666666666666"),
                cooperativa_id=cooperativa_id,
                material_id=mat_pet_id,
                nome_categoria="Garrafa PET",
                categoria_pai_id=cat_plastico_id,
                nome_categoria_pai="Plástico",
                triagem_id=None,
                item_id=None,
                quantidade_kg=Decimal("300.0"),
                tipo_movimentacao="ENTRADA",
                data_movimentacao=datetime(2026, 9, 21, 12, 30, 0),
            ),
        ],
    )

    return {
        "configurable": {
            "snapshots": {
                "material_snapshot": material_snapshot,
                "estoque_snapshot": estoque_snapshot,
                "movimentacao_estoque_snapshot": movimentacao_snapshot,
            }
        }
    }


def test_consultar_estoque_consulta_aberta_retorna_todos_os_materiais(toolkit, mock_snapshots):
    """Consulta aberta (sem material_ids nem material_nomes) deve incluir todos os materiais
    do estoque, mesmo que não estejam com preços cadastrados no material_snapshot."""
    res = toolkit.consultar_estoque(config=mock_snapshots)

    assert res["status"] == "ok"
    assert res["count"] == 2
    assert "Latinha" in res["quantidade_por_categoria_kg"]
    assert "Garrafa PET" in res["quantidade_por_categoria_kg"]
    assert res["quantidade_por_categoria_kg"]["Latinha"] == "150.0"
    assert res["quantidade_por_categoria_kg"]["Garrafa PET"] == "300.0"


def test_consultar_estoque_com_filtro_por_nome_plural(toolkit, mock_snapshots):
    """Verifica correspondência tolerante a plurais (ex: 'latinhas' -> 'Latinha')."""
    res = toolkit.consultar_estoque(config=mock_snapshots, material_nomes=["latinhas"])

    assert res["status"] == "ok"
    assert res["count"] == 1
    assert res["itens"][0]["nome_categoria"] == "Latinha"


def test_generalizar_estoque_consulta_aberta(toolkit, mock_snapshots):
    """Verifica agrupamento por categoria-pai sem filtro de material."""
    res = toolkit.generalizar_estoque(config=mock_snapshots)

    assert res["status"] == "ok"
    assert "Metal" in res["quantidade_por_categoria_pai_kg"]
    assert "Plástico" in res["quantidade_por_categoria_pai_kg"]


def test_buscar_estoque_historico_consulta_aberta(toolkit, mock_snapshots):
    """Verifica busca histórica aberta contendo todas as movimentações."""
    res = toolkit.buscar_estoque_historico(
        config=mock_snapshots,
        data_inicio=datetime(2026, 9, 1),
        data_fim=datetime(2026, 9, 30),
    )

    assert res["status"] == "ok"
    assert res["count"] == 2
    assert "Latinha" in res["quantidade_por_categoria_kg"]
    assert "Garrafa PET" in res["quantidade_por_categoria_kg"]


def test_consultar_estoque_granular_retorna_todas_movimentacoes_do_periodo(toolkit, mock_snapshots):
    """Verifica que consultar_estoque_granular traz tanto Latinha quanto Garrafa PET."""
    res = toolkit.consultar_estoque_granular(
        config=mock_snapshots,
        data_referencia=datetime(2026, 9, 21),
        granularidade="ANO",
    )

    assert res["status"] == "ok"
    assert res["count"] == 2
    assert "Latinha" in res["quantidade_por_categoria_kg"]
    assert "Garrafa PET" in res["quantidade_por_categoria_kg"]
    assert res["quantidade_por_tipo_movimentacao_kg"]["ENTRADA"] == "450.0"


def test_filtro_material_inexistente_retorna_erro(toolkit, mock_snapshots):
    """Se filtrar por um material que não existe nos snapshots, retorna mensagem de erro."""
    res = toolkit.consultar_estoque(config=mock_snapshots, material_nomes=["Cobre"])

    assert res["status"] == "error"
    assert "Nenhum material correspondente encontrado" in res["message"]
