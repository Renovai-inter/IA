from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from typing import List
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, llm: BaseChatModel, system_prompt: str, tools: List[StructuredTool] | None = None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._runnable = create_agent(model=self.llm, tools=self.tools, system_prompt=self.system_prompt)

    def obter_texto_mensagem(self, msg) -> str:
        if hasattr(msg, "content"):
            return msg.content
        elif isinstance(msg, dict):
            return msg.get("content", "")
        return str(msg)
    
    @abstractmethod
    def run(self, state):
        raise NotImplementedError