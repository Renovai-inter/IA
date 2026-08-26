from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from typing import (
    List, Optional
)
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, llm: BaseChatModel, system_prompt: str, tools: List[BaseTool] | None = None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._runnable = create_agent(model=self.llm, tools=self.tools, system_prompt=self.system_prompt)
        
    @abstractmethod
    def run(self, state):
        ...