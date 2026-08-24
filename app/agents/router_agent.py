from app.agents.base import BaseAgent

from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent

class RouterAgent(BaseAgent):
    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

        self._runnable = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
        )

    def run(self, state):
        pass