from app.core.config import Settings
from app.llms.factory import LLMFactory
from app.memory.resumo_service import ResumoService
from app.prompts import (
    ROUTER_PROMPT_COMPLETO,
    MATERIAL_ESTOQUE_PROMPT_COMPLETO,
    ORQUESTRADOR_PROMPT_COMPLETO
)

from app.llms.gemini_provider import GeminiProvider
from app.llms.groq_provider import GroqProvider

from app.agents.base import BaseAgent
from app.agents.router_agent import RouterAgent
from app.agents.material_estoque_agent import MaterialEstoqueAgent
from app.agents.orchestrator_agent import OrchestratorAgent

from app.graph.builder import GraphBuilder
from app.repository.mongodb.db import get_mongo_conn
from app.repository.mongodb.sessao_repository import SessaoRepository
from app.repository.postgresql.db import build_postgres_pool

from app.repository.base import Repository
from app.repository.postgresql.perfil_repository import PerfilRepository
from app.repository.postgresql.material_repository import MaterialRepository
from app.repository.postgresql.estoque_repository import EstoqueRepository
from app.repository.postgresql.movimentacao_estoque_repository import MovimentacaoEstoqueRepository

from app.tools.memory_tools import MemoriaToolkit
from app.tools.material_estoque_tools import MaterialEstoqueToolkit

from typing import Dict
from langgraph.checkpoint.memory import MemorySaver


memoria_toolkit          = MemoriaToolkit()
material_estoque_toolkit = MaterialEstoqueToolkit()

_AGENT_REGISTRY: dict[str, dict] = {
    'router_agent': {
        'cls': RouterAgent,
        'prompt': ROUTER_PROMPT_COMPLETO,
        'tools': memoria_toolkit.get_tools()
    },
    'material_estoque_agent': {
        'cls': MaterialEstoqueAgent,
        'prompt': MATERIAL_ESTOQUE_PROMPT_COMPLETO,
        'tools': memoria_toolkit.get_tools() + material_estoque_toolkit.get_tools()
    },
    'orchestrator_agent': {
        'cls': OrchestratorAgent,
        'prompt': ORQUESTRADOR_PROMPT_COMPLETO,
        'tools': []
    },
}

def build_container(settings: Settings) -> Container:
    pg_pool = build_postgres_pool(settings.DATABASE_URL)
    mongo_conn = get_mongo_conn(settings.MONGODB_URI)

    _REPOSITORIES_MAP = {
        'perfil_repository':               PerfilRepository(db=pg_pool),
        'material_repository':             MaterialRepository(db=pg_pool),
        'estoque_repository':              EstoqueRepository(db=pg_pool),
        'movimentacao_estoque_repository': MovimentacaoEstoqueRepository(db=pg_pool),

        'sessao_repository':               SessaoRepository(db=mongo_conn)
    }

    providers = {
        'GEMINI': GeminiProvider(settings.GEMINI_API_KEY),
        'GROQ':   GroqProvider(settings.GROQ_API_KEY),
    }
    factory = LLMFactory(providers)

    resumo_service = ResumoService(factory, *settings.AGENT_LLM_MAP['resumo_agent'])

    agents = {}
    for name, spec in _AGENT_REGISTRY.items():
        provider_name, tier = settings.AGENT_LLM_MAP[name]
        llm = factory.get(provider_name, tier)
        agents[name] = spec['cls'](llm=llm, system_prompt=spec['prompt'], tools=spec['tools'])

    graph = GraphBuilder(agents, _REPOSITORIES_MAP, MemorySaver()).build_graph()

    return Container(graph=graph, agentes=agents, pg_pool=pg_pool, repositories=_REPOSITORIES_MAP, resumo_service = resumo_service)


class Container:
    def __init__(self, graph, agentes: Dict[str, BaseAgent], pg_pool, repositories: Dict[str, Repository], resumo_service: ResumoService):
        self.graph = graph
        self.agentes = agentes
        self.pg_pool = pg_pool
        self.repositories = repositories
        self.resumo_service = resumo_service