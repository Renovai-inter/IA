from app.graph.state import GraphState
from app.core.config import Settings
from app.agents.base import BaseAgent
from app.repository.base import Repository

from app.graph.nodes import (
    material_estoque_node,
)

from typing import Dict, List
from functools import partial
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

        agent_repos_map: Dict[str, List[Repository]] = {
            agent_name: [self.repositories[repo_name] for repo_name in repo_names if repo_name in self.repositories]
            for agent_name, repo_names in settings.AGENT_REPOSITORY_MAP.items()
        }

        graph.add_node('router_agent', self.agentes['router_agent'].run)
        graph.add_node(
            'material_estoque_agent', 
            partial(
                material_estoque_node, 
                agent=self.agentes['material_estoque_agent'], 
                repositories=agent_repos_map.get('material_estoque_agent', [])
            )
        )

        graph.set_entry_point('router_agent')
        graph.add_conditional_edges(
            "router_agent",
            lambda state: state["proximo_agente"],
            {"material_estoque_agent": "material_estoque_agent"},
        )

        return graph.compile(checkpointer=self.checkpointer)