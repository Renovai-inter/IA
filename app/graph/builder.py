from app.core.config import Settings
from app.graph.state import GraphState
from app.agents.base import BaseAgent
from app.repository.base import Repository

from app.graph.nodes import make_repo_backed_node

from typing import Dict, List
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import (
    StateGraph,
    END,
)

class GraphBuilder:
    def __init__(self, agentes: Dict[str, BaseAgent], settings: Settings, repositories: Dict[str, Repository], checkpointer):
        self.agentes = agentes
        self.settings = settings
        self.repositories = repositories
        self.checkpointer = checkpointer

    def _decisao_roteador(self, state: GraphState):
        """Lê o protocolo do roteador e devolve o nome do próximo nó.
        VAI TER QUE MUDAR - roteador deve poder chamar mais de um agente (mudar prompt e run() também)"""
        return self.settings.ROUTE_NODE_MAP.get(state['proximo_agente'], 'fim')

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GraphState)

        for agent_name, agent in self.agentes.items():
            repo_names = self.settings.AGENT_REPOSITORY_MAP.get(agent_name, [])
            if repo_names:
                agent_repos = {nome: self.repositories[nome] for nome in repo_names}
                graph.add_node(agent_name, make_repo_backed_node(agent, agent_repos))
            else:
                graph.add_node(agent_name, agent.run)

        graph.set_entry_point('router_agent')

        graph.add_conditional_edges(
            'router_agent',
            self._decisao_roteador,
            {
                'material_estoque_agent': 'material_estoque_agent',
                'fim': END,
            },
        )

        graph.add_edge('material_estoque_agent', 'orchestrator_agent')
        graph.add_edge('orchestrator_agent', END)

        return graph.compile(checkpointer=self.checkpointer)