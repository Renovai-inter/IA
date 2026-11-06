"""Testes unitários do OpenaiLLMProvider e integração com o core do Renovai."""
from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings
from app.core.observabilidade import Observabilidade, PRECO_POR_MILHAO
from app.llms.factory import LLMFactory
from app.llms.openai_provider import OpenaiLLMProvider, OpenAIProvider


def test_openai_provider_tiers_disponiveis():
    prov = OpenaiLLMProvider(api_key="sk-mock-key")

    # HIGH agora utiliza o gpt-6.1-terra para máxima precisão analítica e tool use (sampling fixo)
    llm_high = prov.get_llm("HIGH")
    assert isinstance(llm_high, ChatOpenAI)
    assert llm_high.model_name == "gpt-6.1-terra"
    assert llm_high.temperature is None

    # MEDIUM utiliza gpt-6-luna para síntese rápida (sampling fixo)
    llm_med = prov.get_llm("MEDIUM")
    assert isinstance(llm_med, ChatOpenAI)
    assert llm_med.model_name == "gpt-6-luna"
    assert llm_med.temperature is None

    # LOW utiliza gpt-6-luna para guardrails e roteamento ultra-rápidos (sampling fixo)
    llm_low = prov.get_llm("LOW")
    assert isinstance(llm_low, ChatOpenAI)
    assert llm_low.model_name == "gpt-6-luna"
    assert llm_low.temperature is None

    embed = prov.get_llm("EMBEDDING")
    assert isinstance(embed, OpenAIEmbeddings)
    assert embed.model == "text-embedding-3-small"

    assert prov.get_llm("TIER_INEXISTENTE") is None


def test_openai_provider_alias_pascal_case():
    assert OpenAIProvider is OpenaiLLMProvider


def test_openai_provider_overrides():
    prov = OpenaiLLMProvider(api_key="sk-mock-key", model_high="gpt-4o")
    llm_custom = prov.get_llm("HIGH", temperature=0.1, max_tokens=1000)

    assert isinstance(llm_custom, ChatOpenAI)
    assert llm_custom.temperature == 0.1
    assert llm_custom.max_tokens == 1000


def test_openai_provider_overrides_ignora_temperatura_em_modelos_sem_suporte():
    prov = OpenaiLLMProvider(api_key="sk-mock-key", model_high="gpt-6.1-terra")
    llm_custom = prov.get_llm("HIGH", temperature=0.1, max_tokens=1000)

    assert isinstance(llm_custom, ChatOpenAI)
    assert llm_custom.temperature is None
    assert llm_custom.max_tokens == 1000


def test_openai_provider_config_injection():
    # Modelos passados via construtor (padrão de injeção a partir de Settings)
    prov = OpenaiLLMProvider(
        api_key="sk-mock-key",
        model_high="gpt-4o",
        model_medium="gpt-4o-mini",
        model_low="gpt-4o-mini",
        model_embedding="text-embedding-3-large",
    )
    assert prov.get_llm("HIGH").model_name == "gpt-4o"
    assert prov.get_llm("MEDIUM").model_name == "gpt-4o-mini"
    assert prov.get_llm("LOW").model_name == "gpt-4o-mini"
    assert prov.get_llm("EMBEDDING").model == "text-embedding-3-large"


def test_settings_declara_variaveis_openai_com_defaults():
    settings = Settings()
    assert settings.OPENAI_MODEL_HIGH == "gpt-6.1-terra"
    assert settings.OPENAI_MODEL_MEDIUM == "gpt-6-luna"
    assert settings.OPENAI_MODEL_LOW == "gpt-6-luna"
    assert settings.OPENAI_MODEL_EMBEDDING == "text-embedding-3-small"


def test_openai_provider_integracao_factory():
    prov_openai = OpenaiLLMProvider(api_key="sk-mock-openai")
    factory = LLMFactory(providers={"OPENAI": prov_openai})

    llm = factory.get("OPENAI", "HIGH")
    assert isinstance(llm, ChatOpenAI)


def test_openai_provider_observabilidade():
    prov = OpenaiLLMProvider(api_key="sk-mock-key")
    obs = Observabilidade()
    obs.acoplar_provedores({"OPENAI": prov})

    # Verifica se os callbacks foram injetados nos chat models
    for tier in ("HIGH", "MEDIUM", "LOW"):
        llm = prov.get_llm(tier)
        assert any(isinstance(cb, type(obs.coletor)) for cb in (getattr(llm, "callbacks", []) or []))

    # Verifica tabela de precificação FinOps para os modelos OpenAI
    assert "gpt-6.1-terra" in PRECO_POR_MILHAO
    assert "gpt-6-luna" in PRECO_POR_MILHAO
    assert "gpt-4o-mini" in PRECO_POR_MILHAO
    assert "text-embedding-3-small" in PRECO_POR_MILHAO


def test_config_core_usa_openai_por_padrao():
    settings = Settings()
    # Todos os agentes principais devem mapear para o provedor OPENAI
    for agente, (provedor, tier) in settings.AGENT_LLM_MAP.items():
        assert provedor == "OPENAI", f"Agente '{agente}' ainda não está apontando para OPENAI ({provedor})"
        assert tier in ("HIGH", "MEDIUM", "LOW")
