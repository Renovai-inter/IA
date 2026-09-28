"""Fixtures dos testes de integração via Ollama local.

Nada aqui mocka o LLM — os testes deste pacote rodam contra um servidor
Ollama real (ver tests/integration/ollama_helpers.py para a checagem de
disponibilidade e o skip automático quando ele não está no ar).

Deliberadamente NÃO usa app.core.container.build_container nem
app.core.config.Settings: build_container também abre conexões reais com
Postgres/Mongo/Qdrant, o que exigiria subir toda a infra só para testar o
comportamento dos agentes/grafo contra um LLM local. Os repositórios são
substituídos por dublês (tests/integration/fakes.py) — só o LLM é real.
"""
from __future__ import annotations

import pytest

from app.llms.factory import LLMFactory
from app.llms.ollama_provider import OllamaProvider

from tests.integration.ollama_helpers import OLLAMA_BASE_URL


@pytest.fixture(scope="session")
def ollama_provider() -> OllamaProvider:
    return OllamaProvider(base_url=OLLAMA_BASE_URL)


@pytest.fixture(scope="session")
def ollama_llm_factory(ollama_provider: OllamaProvider) -> LLMFactory:
    # Sem fallback: só 'OLLAMA' registrado, de propósito — se um tier falhar,
    # o teste deve estourar claramente, não cair silenciosamente pra outro
    # provider (que aí não estaria mais testando LLM local nenhum).
    return LLMFactory({"OLLAMA": ollama_provider})
