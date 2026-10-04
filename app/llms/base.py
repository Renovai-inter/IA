"""
Módulo base para provedores de Large Language Models (LLMs).

Define a interface abstrata padrão que todos os provedores de modelos de
linguagem (ex: Groq, OpenAI, Anthropic) devem implementar, garantindo o
desacoplamento da infraestrutura de IA do restante do ecossistema.
"""
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List
)
from langchain_core.language_models.chat_models import BaseChatModel

class LLMProvider(ABC):
    """
    Contrato abstrato para geração de texto,
    chamadas estruturadas e execuções síncronas/assíncronas.
    """

    _LLMs: List[Dict[str, BaseChatModel]]

    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def get_llm(self, tier: str, **kwargs) -> BaseChatModel:
        raise NotImplementedError

class LLMService(ABC):
    """
    Contrato base para os serviços que embrulham um client de LLM/embedding
    já construído atrás de uma operação de domínio — `ResumoService`
    (resume conversa) e `EmbeddingService` (vetoriza texto) implementam esse
    contrato.

    Diferente de `LLMProvider` (que devolve o chat model cru por tier, pronto
    pra qualquer agente usar), aqui não existe um método de negócio comum
    forçado entre as subclasses: resumir uma conversa e vetorizar um texto
    têm assinaturas de entrada genuinamente diferentes, e forçar o mesmo
    nome de método pelas duas repetiria o mesmo problema que já apareceu em
    `MemoryStore.recuperar_historico` entre `MongoMemory` e `QdrantMemory`
    (uma assinatura que não serve igualmente pras duas implementações).

    O que este contrato garante é: (1) todo `LLMService` guarda o client já
    construído em `self._llm`, e (2) todo `LLMService` se identifica por
    `nome` — ponto que a instrumentação de custo/latência por chamada de LLM
    (Fase 5 do roadmap) vai precisar pra decorar/logar por serviço sem ter
    que conhecer a classe concreta de cada um.
    """

    def __init__(self, llm):
        self._llm = llm

    @property
    @abstractmethod
    def nome(self) -> str:
        raise NotImplementedError
