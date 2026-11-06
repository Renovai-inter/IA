"""
Provedor de LLM integrado à plataforma OpenAI.

Implementa a interface 'LLMProvider' encapsulando a comunicação de alta 
eficiência e baixa latência com os modelos OpenAI da família GPT-6 (Terra, Luna),
incluindo suporte a chamadas de ferramentas estruturadas, raciocínio rápido para 
nós de roteamento/guardrails e contingência no grafo multi-agente.

Classes:
    OpenaiLLMProvider: Implementação concreta de 'LLMProvider' para a API OpenAI.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.llms.base import LLMProvider
from app.llms.tuning import kwargs_suportados


def modelo_suporta_temperatura(model: Any) -> bool:
    """Verifica se o modelo da OpenAI aceita ajuste customizado de temperatura e top_p.

    Modelos da família GPT-6 (Terra, Luna) e modelos de raciocínio (o1, o3, etc.)
    possuem amostragem fixa ou retornam HTTP 400 (BadRequestError) quando recebem
    parâmetros de sampling customizados como 'temperature' ou 'top_p'.
    """
    if isinstance(model, str):
        nome = model.lower().strip()
        if any(nome.startswith(prefix) for prefix in ('gpt-6', 'o1', 'o3', 'o4')):
            return False
        return True

    profile = getattr(model, 'profile', None)
    if isinstance(profile, dict) and profile.get('temperature') is False:
        return False

    model_name = getattr(model, 'model_name', getattr(model, 'model', ''))
    if model_name:
        nome = str(model_name).lower().strip()
        if any(nome.startswith(prefix) for prefix in ('gpt-6', 'o1', 'o3', 'o4')):
            return False

    return True


class OpenaiLLMProvider(LLMProvider):
    """Provedor para a API da OpenAI.
    
    Tiers alinhados à arquitetura do Renovai:
    - HIGH: Agentes especialistas (coleta, financeiro, material_estoque, judge) usando gpt-6.1-terra.
    - MEDIUM: Orquestrador e sínteses intermediárias usando gpt-6-luna.
    - LOW: Guardrails de entrada/saída, roteamento e resumos rápidos usando gpt-6-luna.
    - EMBEDDING: Geração vetorial para FAQ e memórias usando text-embedding-3-small.
    """

    def __init__(
        self,
        api_key: str,
        model_high: str = 'gpt-6.1-terra',
        model_medium: str = 'gpt-6-luna',
        model_low: str = 'gpt-6-luna',
        model_embedding: str = 'text-embedding-3-small',
    ):
        super().__init__(api_key)
        self.model_high = model_high
        self.model_medium = model_medium
        self.model_low = model_low
        self.model_embedding = model_embedding

        self._LLMs = {
            # HIGH = especialistas em tarefas analíticas, banco de dados e auditoria factual
            'HIGH': ChatOpenAI(
                model=self.model_high,
                api_key=self.api_key,
                **kwargs_suportados(
                    ChatOpenAI,
                    reasoning_effort='medium',
                    timeout=10.0,
                    max_retries=0,
                ),
            ),
            # MEDIUM = orquestrador e síntese de respostas entre múltiplos agentes
            'MEDIUM': ChatOpenAI(
                model=self.model_medium,
                api_key=self.api_key,
                **kwargs_suportados(
                    ChatOpenAI,
                    reasoning_effort='low',
                    timeout=8.0,
                    max_retries=0,
                ),
            ),
            # LOW = router, guardrails e resumo: latência mínima e saída determinística
            'LOW': ChatOpenAI(
                model=self.model_low,
                api_key=self.api_key,
                **kwargs_suportados(
                    ChatOpenAI,
                    reasoning_effort='low',
                    max_tokens=512,
                    timeout=5.0,
                    max_retries=0,
                ),
            ),
            # EMBEDDING = geração vetorial para FAQ e memórias
            'EMBEDDING': OpenAIEmbeddings(
                model=self.model_embedding,
                api_key=self.api_key,
            ),
        }

    def get_llm(self, tier: str, **overrides) -> Any:
        """Retorna uma instância de LLM ou Embedding configurada para o tier informado."""
        base_llm = self._LLMs.get(tier)

        if base_llm is None:
            return None
        if overrides:
            limpos = dict(overrides)
            target_model = limpos.get('model') or getattr(base_llm, 'model_name', getattr(base_llm, 'model', ''))
            if not modelo_suporta_temperatura(target_model):
                limpos.pop('temperature', None)
                limpos.pop('top_p', None)
            if not limpos:
                return base_llm
            return base_llm.model_copy(update=limpos)
        return base_llm


# Alias para convenção de nomenclatura PascalCase padrão
OpenAIProvider = OpenaiLLMProvider
