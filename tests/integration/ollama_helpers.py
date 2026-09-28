"""Helpers de disponibilidade do Ollama local — separado de conftest.py pra
poder ser importado diretamente pelos módulos de teste
(`from tests.integration.ollama_helpers import requires_ollama`) sem
depender de como o pytest carrega conftest.py internamente.
"""
from __future__ import annotations

import os

import httpx
import pytest

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def ollama_esta_no_ar(base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


# Avaliado uma vez, na coleta dos testes. Se o Ollama não estiver no ar,
# toda a suíte de integração é pulada (não falha o pytest) — rode
# `ollama serve` e baixe os modelos configurados em
# app/llms/ollama_provider.py (`ollama pull <modelo>`) antes de rodar isto.
requires_ollama = pytest.mark.skipif(
    not ollama_esta_no_ar(),
    reason=(
        f"Nenhum servidor Ollama respondendo em {OLLAMA_BASE_URL}. "
        "Rode `ollama serve` e `ollama pull <modelo>` (ver app/llms/ollama_provider.py) "
        "antes de rodar tests/integration/. Para apontar pra outro host/porta, "
        "defina a env var OLLAMA_BASE_URL."
    ),
)