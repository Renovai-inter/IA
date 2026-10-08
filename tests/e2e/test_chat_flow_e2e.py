"""Teste e2e de verdade: a mesma FastAPI de produção (mesmos routers) +
Postgres/Mongo/Qdrant REAIS do seu .env + LLM local via Ollama.

Cria e apaga seus próprios dados (tests/e2e/seed.py, fixture
`cenario_estoque`) — não deveria colidir com nada que já existe, mas roda
contra bancos de produção, então só dispara com E2E_CONFIRM=1 e Ollama no
ar (ver tests/e2e/e2e_helpers.py e tests/e2e/README.md).
"""
from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from tests.e2e.seed import CenarioE2E


def test_chat_estoque_ponta_a_ponta(e2e_client: TestClient, cenario_estoque: CenarioE2E):
    resposta = e2e_client.post(
        "/chat",
        json={
            "session_id": str(uuid4()),
            "perfil_id": str(cenario_estoque.perfil_id),
            "pergunta": "quanto temos em estoque?",
        },
    )

    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()

    assert corpo["resposta"].strip()
    assert "material_estoque" in corpo["agentes_chamados"]
    assert "orchestrator" in corpo["agentes_chamados"]
