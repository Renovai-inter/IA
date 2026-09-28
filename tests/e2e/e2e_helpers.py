"""Guarda de segurança dos testes e2e.

tests/e2e usa as conexões REAIS de Postgres/Mongo/Qdrant do seu .env — não
sobe nenhum banco isolado (a pedido, pra não complicar). Isso é
deliberadamente colocado atrás de uma env var explícita, pra ninguém rodar
`pytest` na raiz do projeto e acabar escrevendo em produção sem querer.
"""
from __future__ import annotations

import os

import pytest

E2E_CONFIRM = os.environ.get("E2E_CONFIRM") == "1"

requires_e2e_confirmation = pytest.mark.skipif(
    not E2E_CONFIRM,
    reason=(
        "tests/e2e roda contra os bancos REAIS do seu .env (DATABASE_URL / "
        "MONGODB_URI / QDRANT_URL) — sem infra isolada, por decisão sua. "
        "Defina E2E_CONFIRM=1 explicitamente antes de rodar esta suíte, "
        "assim ela nunca dispara sem querer junto de `pytest` na raiz do "
        "projeto (tests/unit e tests/integration continuam rodando normalmente)."
    ),
)
