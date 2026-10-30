"""
Provedor de LLM integrado aos modelos Google Gemini.

Implementa a interface 'LLMProvider' para os modelos da família Gemini, 
gerenciando chamadas assíncronas, formatação de mensagens, chamadas de 
funções (tool calling) e parse das respostas estruturadas da SDK do Google.

Classes:
    GeminiProvider: Implementação concreta de 'LLMProvider' para a API Google Gemini.
"""
from app.llms.base import LLMProvider

import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.llms.tuning import kwargs_suportados

# Suprime notificação de Automatic Function Calling (AFC) do SDK google-genai.
# No LangGraph/LangChain, o loop ReAct e a execução das tools são geridos
# pelo próprio grafo da aplicação, dispensando o AFC proprietário da SDK.
try:
    from google.genai.models import AsyncModels, Models
    AsyncModels._logged_afc_warning = True
    Models._logged_afc_warning = True
except (ImportError, AttributeError):
    pass


class _SilenciarAFCFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return 'automatic function calling' not in msg and 'AFC' not in msg


logging.getLogger('google_genai.models').addFilter(_SilenciarAFCFilter())
logging.getLogger('google_genai').addFilter(_SilenciarAFCFilter())


class GeminiProvider(LLMProvider):
    def __init__(self, api_key):
        super().__init__(api_key)
        self._LLMs = {
            'HIGH': ChatGoogleGenerativeAI(
                model='gemini-3.5-flash-lite',
                api_key=self.api_key,
                **kwargs_suportados(
                    ChatGoogleGenerativeAI,
                    thinking_level='low',
                    max_retries=0,
                    timeout=5.0,
                ),
            ),
            'MEDIUM': ChatGoogleGenerativeAI(
                model='gemini-3.5-flash',
                temperature=0.7,
                top_p=0.95,
                api_key=self.api_key,
                **kwargs_suportados(
                    ChatGoogleGenerativeAI,
                    max_retries=0,
                    timeout=5.0,
                ),
            ),
            'EMBEDDING': GoogleGenerativeAIEmbeddings(
                model='gemini-embedding-2-preview',
                google_api_key=self.api_key,
            )
        }

    def get_llm(self, tier, **overrides):
        """
        Retorna um llm do tier informado.
        """
        base_llm = self._LLMs.get(tier)

        if overrides:
            return base_llm.model_copy(update=overrides)
        return base_llm
