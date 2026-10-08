"""Repositórios falsos para os testes de integração — mesma interface dos
repositórios reais (`get_snapshot(cooperativa_id) -> Snapshot`), mas sem
tocar em Postgres. Servem pra alimentar o grafo com dados determinísticos
enquanto o LLM (Ollama local) é o único componente "de verdade" sob teste.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from app.repository.postgresql.estoque_repository import Estoque, EstoqueSnapshot
from app.repository.postgresql.material_repository import Material, MaterialSnapshot
from app.repository.postgresql.movimentacao_estoque_repository import (
    MovimentacaoEstoque,
    MovimentacaoEstoqueSnapshot,
)


class FakeMaterialRepository:
    def __init__(self, materiais: list[Material]):
        self._materiais = materiais

    def get_snapshot(self, cooperativa_id: UUID) -> MaterialSnapshot:
        return MaterialSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(timezone.utc),
            itens=self._materiais,
        )


class FakeEstoqueRepository:
    def __init__(self, itens: list[Estoque]):
        self._itens = itens

    def get_snapshot(self, cooperativa_id: UUID) -> EstoqueSnapshot:
        return EstoqueSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(timezone.utc),
            itens=self._itens,
        )


class FakeMovimentacaoEstoqueRepository:
    def __init__(self, itens: list[MovimentacaoEstoque] | None = None):
        self._itens = itens or []

    def get_snapshot(self, cooperativa_id: UUID) -> MovimentacaoEstoqueSnapshot:
        return MovimentacaoEstoqueSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(timezone.utc),
            itens=self._itens,
        )


def cenario_papelao(cooperativa_id: UUID) -> dict[str, object]:
    """Um material (papelão) com 3.450 kg em estoque numa cooperativa fictícia.

    Devolve o dict {nome_do_repository: repo_falso} no formato que
    GraphBuilder/make_repo_backed_node esperam (mesmas chaves de
    Settings.AGENT_REPOSITORY_MAP['material_estoque_agent']).
    """
    material_id = UUID(int=1)
    categoria_id = UUID(int=2)

    material = Material(
        material_id=material_id,
        categoria_id=categoria_id,
        nome_categoria="Papelão",
        categoria_pai_id=None,
        nome_categoria_pai=None,
        cooperativa_id=cooperativa_id,
        nome_cooperativa="Cooperativa Teste",
        preco_sugerido=Decimal("0.45"),
        esta_disponivel=True,
    )
    estoque = Estoque(
        estoque_id=UUID(int=3),
        cooperativa_id=cooperativa_id,
        nome_cooperativa="Cooperativa Teste",
        material_id=material_id,
        nome_categoria="Papelão",
        categoria_pai_id=None,
        nome_categoria_pai=None,
        quantidade_kg=Decimal("3450"),
        data_atualizacao=datetime.now(timezone.utc),
    )

    return {
        "material_repository": FakeMaterialRepository([material]),
        "estoque_repository": FakeEstoqueRepository([estoque]),
        "movimentacao_estoque_repository": FakeMovimentacaoEstoqueRepository([]),
    }