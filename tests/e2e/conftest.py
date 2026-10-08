"""Fixtures dos testes e2e.

Toda a suíte aqui (`pytestmark`) só roda com E2E_CONFIRM=1 (segurança —
ver e2e_helpers.py) e com Ollama no ar (reaproveita o marker de
tests/integration). Sem as duas coisas, os testes são pulados, não falham.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import chat, sessions
from app.core.config import Settings

from tests.e2e.e2e_helpers import requires_e2e_confirmation
from tests.e2e.ollama_container import build_e2e_container
from tests.e2e.seed import CenarioE2E, limpar_cenario, seed_cenario_estoque
from tests.integration.ollama_helpers import requires_ollama

pytestmark = [requires_e2e_confirmation, requires_ollama]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def e2e_app(settings: Settings):
    """A mesma FastAPI de produção (mesmos routers de app/main.py), mas com
    o Container montado por build_e2e_container — só troca Gemini/Groq por
    Ollama. Não usa o lifespan de app/main.py de propósito, pra não tentar
    conectar em Gemini/Groq (que exigiria API key com quota real) só para
    montar o app.
    """
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(sessions.router)

    container = build_e2e_container(settings)
    app.state.container = container

    try:
        yield app
    finally:
        for factory, conn in container.factories.values():
            try:
                factory.close(conn)
            except Exception:
                pass


@pytest.fixture(scope="session")
def e2e_client(e2e_app: FastAPI) -> TestClient:
    return TestClient(e2e_app)


@pytest.fixture
def cenario_estoque(settings: Settings) -> CenarioE2E:
    """Cria 1 cooperativa/perfil/categoria/material/estoque de teste no
    Postgres real e apaga tudo ao final, mesmo se o teste falhar."""
    cenario = seed_cenario_estoque(settings.DATABASE_URL)
    try:
        yield cenario
    finally:
        limpar_cenario(settings.DATABASE_URL, cenario)
