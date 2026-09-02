from app.graph.state import GraphState
from app.core.config import Settings
from app.agents.base import BaseAgent
from app.repository.base import Repository

from app.graph.nodes import make_repo_backed_node

from typing import Dict
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import (
    StateGraph,
    END,
)

class GraphBuilder:
    def __init__(self, agentes: Dict[str, BaseAgent], repositories: Dict[str, Repository], checkpointer):
        self.agentes = agentes
        self.repositories = repositories
        self.checkpointer = checkpointer
        
    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GraphState)
        settings = Settings()

        for agent_name, agent in self.agentes.items():
            repo_names = settings.AGENT_REPOSITORY_MAP.get(agent_name, [])
            if repo_names:
                agent_repos = {nome: self.repositories[nome] for nome in repo_names}
                graph.add_node(agent_name, make_repo_backed_node(agent, agent_repos))
            else:
                graph.add_node(agent_name, agent.run)

        graph.set_entry_point('router_agent')
        graph.add_conditional_edges(
            "router_agent",
            lambda state: state["proximo_agente"],
            {"material_estoque_agent": "material_estoque_agent"},
        )

        return graph.compile(checkpointer=self.checkpointer)