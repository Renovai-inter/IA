"""Smoke test: confirma que os 3 tiers configurados no OllamaProvider
respondem de fato via Ollama local, antes de rodar qualquer coisa mais
pesada (agente, grafo) em cima disso. Se isto falhar, o problema é
infraestrutura local (modelo não baixado, Ollama não rodando, nome de
modelo errado em app/llms/ollama_provider.py) — não os agentes.
"""
from __future__ import annotations

import pytest

from tests.integration.ollama_helpers import check_modelo_ou_skip, requires_ollama


@requires_ollama
@pytest.mark.parametrize("tier", ["HIGH", "MEDIUM", "LOW"])
def test_tier_responde(ollama_provider, tier):
    llm = ollama_provider.get_llm(tier)
    check_modelo_ou_skip(llm, tier=tier)
    resposta = llm.invoke("Responda apenas com a palavra: ok")
    assert resposta.content.strip()