from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from app.graph.state import GraphState

from typing import List
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, llm: BaseChatModel, system_prompt: str, tools: List[StructuredTool] | None = None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._runnable = create_agent(model=self.llm, tools=self.tools, system_prompt=self.system_prompt)

    def _obter_texto_mensagem(self, msg) -> str:
        content = msg.content
        if isinstance(content, str):
            return content
        return "".join(
            bloco.get("text", "")
            for bloco in content
            if isinstance(bloco, dict) and bloco.get("type") == "text"
    )
    
    @abstractmethod
    def run(self, state: GraphState, config: RunnableConfig):
        raise NotImplementedError