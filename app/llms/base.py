"""
Módulo base para provedores de Large Language Models (LLMs).

Define a interface abstrata padrão que todos os provedores de modelos de 
linguagem (ex: Groq, OpenAI, Anthropic) devem implementar, garantindo o
desacoplamento da infraestrutura de IA do restante do ecossistema.
"""
from abc import ABC, abstractmethod
from typing import (
    Any,
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
        ...