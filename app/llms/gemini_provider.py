"""
Provedor de LLM integrado aos modelos Google Gemini.

Implementa a interface 'LLMProvider' para os modelos da família Gemini, 
gerenciando chamadas assíncronas, formatação de mensagens, chamadas de 
funções (tool calling) e parse das respostas estruturadas da SDK do Google.

Classes:
    GeminiProvider: Implementação concreta de 'LLMProvider' para a API Google Gemini.
"""
from app.llms.base import LLMProvider

from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiProvider(LLMProvider):
    def __init__(self, api_key):
        super().__init__(api_key)
        self._LLMs = {
            'HIGH': ChatGoogleGenerativeAI(
                model='gemini-2.5-flash',
                temperature=0.7,
                top_p=0.95,
                api_key=self.api_key
            ),
            'MEDIUM': ChatGoogleGenerativeAI(
                model='gemini-2.5-flash',
                temperature=0.7,
                top_p=0.95,
                api_key=self.api_key
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