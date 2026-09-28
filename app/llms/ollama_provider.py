"""
Provedor de LLM integrado a um servidor Ollama local.

Implementa a interface 'LLMProvider' para modelos rodando localmente via
Ollama (https://ollama.com), seguindo o mesmo padrão de 'GeminiProvider' e
'GroqProvider'. Existe principalmente para permitir rodar agentes e o grafo
completo em testes (ver tests/integration/) sem consumir quota de API nem
depender de rede — não é usado no container de produção (app/core/container.py)
a menos que você registre explicitamente 'OLLAMA' no dicionário de providers.

Pré-requisitos:
    - Ollama instalado e rodando localmente (`ollama serve`).
    - Os modelos abaixo baixados localmente (`ollama pull <modelo>`), ou os
      nomes trocados para modelos que você já tenha.
    - Dependência 'langchain-ollama' instalada (ver pyproject.toml).

Classes:
    OllamaProvider: Implementação concreta de 'LLMProvider' para Ollama local.
"""
from app.llms.base import LLMProvider

from langchain_ollama import ChatOllama


class OllamaProvider(LLMProvider):
    """
    Tiers pensados pra espelhar os tiers já usados em GeminiProvider/GroqProvider
    (HIGH = agentes especialistas, LOW = roteador/classificadores rápidos).
    Os nomes de modelo abaixo são um ponto de partida razoável para tool-calling
    em Ollama — troque pelos que você já tem baixados localmente se preferir.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(api_key=None)
        self.base_url = base_url
        self._LLMs = {
            'HIGH': ChatOllama(
                model='qwen2.5:14b-instruct',
                base_url=self.base_url,
                temperature=0.7,
            ),
            'MEDIUM': ChatOllama(
                model='qwen2.5:7b-instruct',
                base_url=self.base_url,
                temperature=0.7,
            ),
            'LOW': ChatOllama(
                model='qwen2.5:3b-instruct',
                base_url=self.base_url,
                temperature=0.0,
            ),
        }

    def get_llm(self, tier, **overrides):
        """
        Retorna um llm do tier informado.
        """
        base_llm = self._LLMs.get(tier)

        if overrides:
            return base_llm.model_copy(update=overrides)
        return base_llm