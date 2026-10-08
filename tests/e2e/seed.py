"""Seed/cleanup de um cenário mínimo direto no Postgres real (via psycopg),
usado pelos testes e2e.

Não passa pelos Repository (eles só leem) — insere e apaga com SQL simples,
sempre dentro de um cenário isolado (UUIDs próprios, nomes marcados com
"[E2E PYTEST ...]") pra nunca colidir com dados reais e sempre dar pra
limpar mesmo se um teste falhar no meio (o fixture em conftest.py chama
`limpar_cenario` dentro de um `finally`).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row


@dataclass
class CenarioE2E:
    cooperativa_id: UUID
    perfil_id: UUID
    categoria_id: UUID
    material_id: UUID
    estoque_id: UUID


def seed_cenario_estoque(dsn: str, quantidade_kg: Decimal = Decimal("3450")) -> CenarioE2E:
    """Cria: 1 cooperativa, 1 perfil (tipo COOPERATIVA), 1 categoria de
    material, 1 material e 1 linha de estoque — o mínimo pro fluxo
    router -> material_estoque_agent -> orchestrator ter o que consultar.
    """
    marca = f"[E2E PYTEST {uuid4().hex[:8]}]"

    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cooperativas (nome, numero_cooperados)
                VALUES (%s, 1)
                RETURNING cooperativa_id
                """,
                (f"Cooperativa {marca}",),
            )
            cooperativa_id = cur.fetchone()["cooperativa_id"]

            cur.execute(
                """
                INSERT INTO perfis (cooperativa_id, email, cnpj)
                VALUES (%s, %s, %s)
                RETURNING perfil_id
                """,
                (
                    cooperativa_id,
                    f"e2e.{uuid4().hex[:12]}@teste.invalido",
                    f"00.000.000/0001-{uuid4().int % 100:02d}",
                ),
            )
            perfil_id = cur.fetchone()["perfil_id"]

            cur.execute(
                """
                INSERT INTO categorias_materiais (nome_categoria)
                VALUES (%s)
                RETURNING categoria_id
                """,
                (f"Papelão {marca}",),
            )
            categoria_id = cur.fetchone()["categoria_id"]

            cur.execute(
                """
                INSERT INTO materiais (categoria_id, cooperativa_id, preco_sugerido, esta_disponivel)
                VALUES (%s, %s, %s, TRUE)
                RETURNING material_id
                """,
                (categoria_id, cooperativa_id, Decimal("0.45")),
            )
            material_id = cur.fetchone()["material_id"]

            cur.execute(
                """
                INSERT INTO estoques (cooperativa_id, material_id, quantidade_kg)
                VALUES (%s, %s, %s)
                RETURNING estoque_id
                """,
                (cooperativa_id, material_id, quantidade_kg),
            )
            estoque_id = cur.fetchone()["estoque_id"]

    return CenarioE2E(
        cooperativa_id=cooperativa_id,
        perfil_id=perfil_id,
        categoria_id=categoria_id,
        material_id=material_id,
        estoque_id=estoque_id,
    )


def limpar_cenario(dsn: str, cenario: CenarioE2E) -> None:
    """Apaga só as linhas que `seed_cenario_estoque` criou, na ordem que
    respeita as dependências (estoque/movimentação antes de material,
    material antes de categoria, perfil antes de cooperativa)."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM movimentacoes_estoques WHERE estoque_id = %s", (cenario.estoque_id,))
            cur.execute("DELETE FROM estoques WHERE estoque_id = %s", (cenario.estoque_id,))
            cur.execute("DELETE FROM materiais WHERE material_id = %s", (cenario.material_id,))
            cur.execute("DELETE FROM categorias_materiais WHERE categoria_id = %s", (cenario.categoria_id,))
            cur.execute("DELETE FROM perfis WHERE perfil_id = %s", (cenario.perfil_id,))
            cur.execute("DELETE FROM cooperativas WHERE cooperativa_id = %s", (cenario.cooperativa_id,))
