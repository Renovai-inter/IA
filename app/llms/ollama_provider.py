"""
Provedor de LLM integrado a um servidor Ollama local.

Implementa a interface 'LLMProvider' para modelos rodando localmente via
Ollama (https://ollama.com), seguindo o mesmo padrão de 'GeminiProvider' e
'GroqProvider'. Existe principalmente para permitir rodar agentes e o grafo
completo em testes (ver tests/integration/) sem consumir quota de API nem
depender de rede — não é usado no container de produção (app/core/container.py)
a menos que você registre explicitamente 'OLLAMA' no dicionário de providers.

Configuração via variáveis de ambiente:
    - OLLAMA_BASE_URL: URL do servidor Ollama (default: http://localhost:11434).
    - OLLAMA_TIMEOUT: Tempo limite em segundos para requisições (default: 120.0).
    - OLLAMA_MODEL: Define o mesmo modelo para todos os tiers (ex: qwen2.5:3b-instruct).
    - OLLAMA_MODEL_HIGH: Modelo específico do tier HIGH (default: qwen2.5:3b-instruct ou disponível).
    - OLLAMA_MODEL_MEDIUM: Modelo específico do tier MEDIUM (default: qwen2.5:3b-instruct ou disponível).
    - OLLAMA_MODEL_LOW: Modelo específico do tier LOW (default: qwen2.5:3b-instruct ou disponível).

Classes:
    OllamaProvider: Implementação concreta de 'LLMProvider' para Ollama local.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_ollama import ChatOllama

from app.llms.base import LLMProvider


def _obter_modelos_instalados(base_url: str) -> list[str]:
    """Obtém os modelos de chat instalados localmente no Ollama."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            names = [m.get("name", "") for m in models if m.get("name")]
            return [m for m in names if "embed" not in m.lower() and "bge" not in m.lower()]
    except Exception:
        pass
    return []


def _normalizar_modelo(nome: str) -> str:
    nome = nome.lower().strip()
    if nome.endswith(":latest"):
        nome = nome[:-7]
    return nome


def _resolver_modelo(
    modelo_desejado: str,
    override: str | None,
    instalados: list[str],
) -> str:
    """Define o modelo efetivo: override explícito > correspondência instalada > primeiro instalado > desejado."""
    if override:
        return override

    alvo = _normalizar_modelo(modelo_desejado)
    for inst in instalados:
        inst_norm = _normalizar_modelo(inst)
        if inst_norm == alvo or inst_norm.startswith(alvo) or alvo.startswith(inst_norm):
            return inst

    if instalados:
        return instalados[0]

    return modelo_desejado


class OllamaProvider(LLMProvider):
    """
    Tiers pensados pra espelhar os tiers já usados em GeminiProvider/GroqProvider
    (HIGH = agentes especialistas, LOW = roteador/classificadores rápidos).
    Usa por padrão modelos leves (3B) para viabilizar testes locais rápidos sem travamento,
    com suporte a timeout e detecção de modelos já baixados.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_high: str | None = None,
        model_medium: str | None = None,
        model_low: str | None = None,
        timeout: float | None = None,
    ):
        super().__init__(api_key=None)
        self.base_url = os.environ.get("OLLAMA_BASE_URL", base_url)

        if timeout is not None:
            self.timeout = timeout
        else:
            env_timeout = os.environ.get("OLLAMA_TIMEOUT")
            self.timeout = float(env_timeout) if env_timeout else 120.0

        fallback_global = os.environ.get("OLLAMA_MODEL")
        override_high = model_high or os.environ.get("OLLAMA_MODEL_HIGH") or fallback_global
        override_med = model_medium or os.environ.get("OLLAMA_MODEL_MEDIUM") or fallback_global
        override_low = model_low or os.environ.get("OLLAMA_MODEL_LOW") or fallback_global

        instalados = _obter_modelos_instalados(self.base_url)

        m_high = _resolver_modelo("qwen2.5:3b-instruct", override_high, instalados)
        m_med = _resolver_modelo("qwen2.5:3b-instruct", override_med, instalados)
        m_low = _resolver_modelo("qwen2.5:3b-instruct", override_low, instalados)

        client_kwargs = {"timeout": self.timeout}

        self._LLMs = {
            'HIGH': ChatOllama(
                model=m_high,
                base_url=self.base_url,
                temperature=0.7,
                client_kwargs=client_kwargs,
            ),
            'MEDIUM': ChatOllama(
                model=m_med,
                base_url=self.base_url,
                temperature=0.7,
                client_kwargs=client_kwargs,
            ),
            'LOW': ChatOllama(
                model=m_low,
                base_url=self.base_url,
                temperature=0.0,
                client_kwargs=client_kwargs,
            ),
        }

    def get_llm(self, tier: str, **overrides) -> Any:
        """
        Retorna um llm do tier informado.
        """
        base_llm = self._LLMs.get(tier)
        if base_llm is None:
            raise ValueError(f"Tier desconhecido '{tier}'. Tiers válidos: {list(self._LLMs.keys())}")

        if overrides:
            return base_llm.model_copy(update=overrides)
        return base_llm