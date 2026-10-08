"""Container para os testes e2e utilizando Ollama local.

Reutiliza o mesmo `build_container` de produção (app/core/container.py),
substituindo apenas os provedores de LLM por `OllamaProvider` e mapeando
todos os agentes para o provider 'OLLAMA'.
Postgres, Mongo e Qdrant continuam sendo os bancos reais do seu .env.
"""
from __future__ import annotations

from app.core.config import Settings
from app.core.container import Container, build_container
from app.llms.ollama_provider import OllamaProvider


_OLLAMA_AGENT_LLM_MAP: dict[str, tuple[str, str]] = {
    "router_agent": ("OLLAMA", "LOW"),
    "faq_agent": ("OLLAMA", "MEDIUM"),
    "material_estoque_agent": ("OLLAMA", "HIGH"),
    "orchestrator_agent": ("OLLAMA", "LOW"),
    "resumo_agent": ("OLLAMA", "LOW"),
}


def build_e2e_container(settings: Settings) -> Container:
    providers = {"OLLAMA": OllamaProvider(base_url=settings.OLLAMA_BASE_URL)}
    return build_container(settings, providers=providers, agent_llm_map=_OLLAMA_AGENT_LLM_MAP)
