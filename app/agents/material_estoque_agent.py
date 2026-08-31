from app.agents.base import BaseAgent
from app.graph.state import GraphState

from langchain.agents import create_agent

class RouterAgent(BaseAgent):
    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

        self._runnable = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
        )

    def run(self, state: GraphState) -> dict:
        resultado = self._runnable.invoke(
            {"messages": list(state["messages"])},
            config={"configurable": {"estoque_snapshot": state["estoque_snapshot"]}},
        )