"""
Provedor de LLM integrado à plataforma Groq.

Implementa a interface 'LLMProvider' encapsulando a comunicação de ultra-baixa 
latência com a API da Groq (Llama, Mixtral, etc.), incluindo estratégias 
de fallback, retry e injeção de configurações de inferência.

Classes:
    GroqProvider: Implementação concreta de 'LLMProvider' para a API Groq.
"""

from llms.base import LLMProvider

from langchain_groq import ChatGroq

class GroqProvider(LLMProvider):
    def __init__(self, api_key):
        super().__init__(api_key)
        self._LLMs = [
            {
                'HIGH' : ChatGroq(
                    model='qwen/qwen3.6-27b',
                    temperature=0.7,
                    api_key=self.api_key
                ),
                'MEDIUM' : ChatGroq(
                    model='openai/gpt-oss-120b',
                    temperature=0.7,
                    api_key=self.api_key
                ),
                'LOW' : ChatGroq(
                    model='openai/gpt-oss-20b',
                    temperature=0.0,
                    api_key=self.api_key
                ),
            }
        ]

    def get_llm(self, tier, **overrides):
        """
        Retorna um llm do tier informado.

        Kwargs é opcional, quando passado retorna uma nova instância 
        de ChatGroq com os argumentos fornecidos.
        """
        base_llm = self._LLMs[tier]

        if overrides:
            return base_llm.model_copy(update=overrides)
        return base_llm