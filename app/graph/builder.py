from app.graph.state import GraphState
from app.agents.base import BaseAgent

from typing import Dict
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import (
    StateGraph,
    END,
)

class GraphBuilder():
    def __init__(self, agentes: Dict[str, BaseAgent], checkpointer):
        self.agentes = agentes
        self.checkpointer = checkpointer
        
    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GraphState)
        graph.add_node('router_agent', self.agentes['router_agent'].run)

        graph.set_entry_point('router_agent')
        graph.add_edge('router_agent', END)

        return graph.compile(checkpointer=self.checkpointer)