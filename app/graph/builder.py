from app.graph.registro import RegistroEspecialista
from app.graph.state import GraphState
from app.agents.base import BaseAgent
from app.repository.base import Repository

from app.graph.nodes import make_repo_backed_node

from typing import Dict, List, Optional
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
import time, functools, logging


class GraphBuilder:
    def __init__(
        self,
        router_agent: BaseAgent,
        orchestrator_agent: BaseAgent,
        specialists: List[RegistroEspecialista],
        repositories: Dict[str, Repository],
        checkpointer: BaseCheckpointSaver,
        guardrail_entrada: BaseAgent,
        guardrail_saida: BaseAgent,
    ):
        self._router_agent = router_agent
        self._orchestrator_agent = orchestrator_agent
        self._specialists = specialists
        self._repositories = repositories
        self._checkpointer = checkpointer
        self._guardrail_entrada = guardrail_entrada
        self._guardrail_saida = guardrail_saida

        self.log = logging.getLogger("latency")


    def timed(self, nome):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                t = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    self.log.warning("node=%s %.2fs", nome, time.perf_counter() - t)
            return wrapper
        return deco

    # --------------------------------------------------------------------------
    # Decisões
    # --------------------------------------------------------------------------
    def _decisao_guardrail_entrada(self, state: GraphState) -> str:
        return 'router_agent' if state.get('passou_guardrail') else 'fim'

    def _decisao_roteador(self, state: GraphState) -> str:
        """Lê o protocolo do roteador e devolve o nome do próximo nó.
        VAI TER QUE MUDAR - roteador deve poder chamar mais de um agente (mudar prompt e run() também)"""
        proximo = state.get('proximo_agente')
        rotas_especialistas = {registro.rota for registro in self._specialists}
        if proximo in rotas_especialistas:
            return proximo
        return 'fim'

    # --------------------------------------------------------------------------
    # Montagem
    # --------------------------------------------------------------------------
    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GraphState)

        graph.add_node('guardrail_entrada', self.timed('guardrail_entrada')(self._guardrail_entrada.run))
        graph.add_node('router_agent', self.timed('router')(self._router_agent.run))

        graph.add_node('orchestrator_agent', self.timed('orchestrator')(self._orchestrator_agent.run))
        graph.add_node('guardrail_saida', self.timed('guardrail_saida')(self._guardrail_saida.run))

        route_node_map: Dict[str, str] = {}
        for registro in self._specialists:
            agent_repos = {nome: self._repositories[nome] for nome in registro.repositories}
            node = make_repo_backed_node(registro.agente, agent_repos)

            graph.add_node(registro.node_name, self.timed(registro.node_name)(node))
            graph.add_edge(registro.node_name, 'orchestrator_agent') # troca por 'juiz' quando ele existir
            route_node_map[registro.rota] = registro.node_name

        # Entrada
        graph.set_entry_point('guardrail_entrada')
        graph.add_conditional_edges(
            'guardrail_entrada',
            self._decisao_guardrail_entrada,
            {'router_agent': 'router_agent', 'fim': END},
        )

        # Roteador
        graph.add_conditional_edges(
            'router_agent',
            self._decisao_roteador,
            {**route_node_map, 'fim': END}
        )

        # Saída
        graph.add_edge('orchestrator_agent', 'guardrail_saida')
        graph.add_edge('guardrail_saida', END)

        return graph.compile(checkpointer=self._checkpointer)