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
    except (httpx.HTTPError, Exception):
        return False


def get_modelos_instalados(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Retorna a lista de modelos baixados no Ollama local."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        pass
    return []


def normalizar_nome_modelo(nome: str) -> str:
    nome = nome.lower().strip()
    if nome.endswith(":latest"):
        nome = nome[:-7]
    return nome


def modelo_disponivel(modelo: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """Verifica se o modelo está baixado no Ollama local."""
    instalados = get_modelos_instalados(base_url)
    if not instalados:
        return False
    alvo = normalizar_nome_modelo(modelo)
    for inst in instalados:
        inst_norm = normalizar_nome_modelo(inst)
        if inst_norm == alvo or inst_norm.startswith(alvo) or alvo.startswith(inst_norm):
            return True
    return False


def check_modelo_ou_skip(llm: object, tier: str = "") -> None:
    """Valida se o modelo configurado no LLM está disponível no Ollama local.
    Se não estiver, pula o teste com uma mensagem clara instruindo o usuário."""
    modelo = getattr(llm, "model", None)
    if not modelo:
        return
    if not modelo_disponivel(modelo):
        instalados = get_modelos_instalados()
        msg_instalados = ", ".join(instalados) if instalados else "nenhum modelo encontrado"
        pytest.skip(
            f"Modelo '{modelo}' ({f'tier {tier}' if tier else 'solicitado'}) não está instalado no Ollama local. "
            f"Modelos disponíveis: [{msg_instalados}]. "
            f"Execute: `ollama pull {modelo}` ou configure via env var (ex: OLLAMA_MODEL_{tier})."
        )


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