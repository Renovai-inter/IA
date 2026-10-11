import inspect, time, functools, logging
from typing import Dict, List, Optional
from langchain_core.runnables import RunnableLambda
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.graph.registro import RegistroEspecialista
from app.graph.state import GraphState
from app.agents.base import BaseAgent
from app.repository.base import Repository
from app.graph.nodes import make_repo_backed_node


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


    def timed(self, nome: str):
        def deco(fn):
            if isinstance(fn, RunnableLambda):
                return RunnableLambda(
                    deco(fn.func) if fn.func else None,
                    afunc=deco(fn.afunc) if fn.afunc else None,
                )
            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*a, **kw):
                    t = time.perf_counter()
                    try:
                        return await fn(*a, **kw)
                    finally:
                        self.log.warning("node=%s %.2fs", nome, time.perf_counter() - t)
                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*a, **kw):
                    t = time.perf_counter()
                    try:
                        return fn(*a, **kw)
                    finally:
                        self.log.warning("node=%s %.2fs", nome, time.perf_counter() - t)
                return sync_wrapper
        return deco

    # --------------------------------------------------------------------------
    # Decisões
    # --------------------------------------------------------------------------
    def _decisao_guardrail_entrada(self, state: GraphState) -> str:
        return 'router_agent' if state.get('passou_guardrail') else 'fim'

    def _decisao_roteador(self, state: GraphState) -> list[str] | str:
        """Lê o protocolo do roteador e devolve a lista de rotas especialistas a invokar.
        Se nenhuma rota mapeada for informada, devolve 'fim'."""
        rotas = state.get('proximos_agentes') or []
        if not rotas:
            return 'fim'

        rotas_especialistas = {registro.rota for registro in self._specialists}
        rotas_validas = [r for r in rotas if r in rotas_especialistas]

        return rotas_validas if rotas_validas else 'fim'

    def _make_agent_node(self, agent: BaseAgent):
        sync_fn = agent.run
        async_fn = getattr(agent, "arun", None)
        if async_fn is not None:
            return RunnableLambda(sync_fn, afunc=async_fn)
        return sync_fn

    # --------------------------------------------------------------------------
    # Montagem
    # --------------------------------------------------------------------------
    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(GraphState)

        graph.add_node('guardrail_entrada', self.timed('guardrail_entrada')(self._make_agent_node(self._guardrail_entrada)))
        graph.add_node('router_agent', self.timed('router')(self._make_agent_node(self._router_agent)))

        graph.add_node('orchestrator_agent', self.timed('orchestrator')(self._make_agent_node(self._orchestrator_agent)))
        graph.add_node('guardrail_saida', self.timed('guardrail_saida')(self._make_agent_node(self._guardrail_saida)))

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