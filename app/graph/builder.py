from app.graph.registro import RegistroEspecialista
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
from langgraph.checkpoint.base import BaseCheckpointSaver


class GraphBuilder:
    def __init__(
        self,
        router_agent: BaseAgent,
        orchestrator_agent: BaseAgent,
        specialists: List[RegistroEspecialista],
        repositories: Dict[str, Repository],
        checkpointer: BaseCheckpointSaver,
    ):
        self._router_agent = router_agent
        self._orchestrator_agent = orchestrator_agent
        self._specialists = specialists
        self._repositories = repositories
        self._checkpointer = checkpointer

    def _decisao_roteador(self, state: GraphState) -> str:
        """Lê o protocolo do roteador e devolve o nome do próximo nó.
        VAI TER QUE MUDAR - roteador deve poder chamar mais de um agente (mudar prompt e run() também)"""
        return state['proximo_agente']

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GraphState)
        graph.add_node('router_agent', self._router_agent.run)
        graph.add_node('orchestrator_agent', self._orchestrator_agent.run)

        route_node_map: Dict[str, str] = {}
        for registro in self._specialists:
            agent_repos = {nome: self._repositories[nome] for nome in registro.repositories}
            node = make_repo_backed_node(registro.agente, agent_repos)

            graph.add_node(registro.node_name, node)
            graph.add_edge(registro.node_name, 'orchestrator_agent') # troca por 'juiz' quando ele existir
            route_node_map[registro.rota] = registro.node_name

        graph.set_entry_point('router_agent')

        graph.add_conditional_edges(
            'router_agent',
            self._decisao_roteador,
            {**route_node_map, 'fim': END}
        )

        graph.add_edge('orchestrator_agent', END)

        return graph.compile(checkpointer=self._checkpointer)