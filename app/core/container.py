from app.core.config import Settings
from app.graph.registro import RegistroEspecialista
from app.llms.factory import LLMFactory
from app.memory.mongo_memory import MongoMemory
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
from app.repository.mongodb.db import MongoConnectionFactory
from app.repository.mongodb.sessao_repository import SessaoRepository
from app.repository.postgresql.db import PostgresConnectionFactory

from app.repository.base import ConnectionFactory, Repository
from app.repository.postgresql.perfil_repository import PerfilRepository
from app.repository.postgresql.material_repository import MaterialRepository
from app.repository.postgresql.estoque_repository import EstoqueRepository
from app.repository.postgresql.movimentacao_estoque_repository import MovimentacaoEstoqueRepository

from app.repository.qdrant.db import QdrantConnectionFactory
from app.tools.memory_tools import MemoriaToolkit
from app.tools.material_estoque_tools import MaterialEstoqueToolkit

from typing import Any, Dict, Tuple
from langgraph.checkpoint.memory import MemorySaver


def build_container(settings: Settings) -> Container:
    pg_factory = PostgresConnectionFactory(dsn=settings.DATABASE_URL)
    mongo_factory = MongoConnectionFactory(dsn=settings.MONGODB_URI, db_name='mongo_dbrenovai')
    qdrant_factory = QdrantConnectionFactory(dsn=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    _FACTORIES_MAP = {
        'pg_factory':     (pg_factory,     pg_factory.connect()),
        'mongo_factory':  (mongo_factory,  mongo_factory.connect()),
        'qdrant_factory': (qdrant_factory, qdrant_factory.connect()),
    }

    _REPOSITORIES_MAP = {
        'perfil_repository':               PerfilRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'material_repository':             MaterialRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'estoque_repository':              EstoqueRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'movimentacao_estoque_repository': MovimentacaoEstoqueRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),

        'sessao_repository':               SessaoRepository(db=_FACTORIES_MAP.get('mongo_factory')[1])
    }

    providers = {
        'GEMINI': GeminiProvider(settings.GEMINI_API_KEY),
        'GROQ':   GroqProvider(settings.GROQ_API_KEY),
    }
    factory = LLMFactory(providers)

    resumo_service = ResumoService(factory, *settings.AGENT_LLM_MAP['resumo_agent'])
    mongo_memory = MongoMemory(_REPOSITORIES_MAP['sessao_repository'], resumo_service)

    memoria_toolkit          = MemoriaToolkit(mongo_memory)
    material_estoque_toolkit = MaterialEstoqueToolkit()

    _AGENT_REGISTRY: dict[str, dict] = {
        'router_agent': {
            'cls': RouterAgent,
            'prompt': ROUTER_PROMPT_COMPLETO,
            'tools': memoria_toolkit.get_tools()
        },
        'material_estoque_agent': {
            'route': 'estoque',
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

    specialists: list[RegistroEspecialista] = []
    agents = {}
    for name, spec in _AGENT_REGISTRY.items():
        llm = factory.get(*settings.AGENT_LLM_MAP[name])
        agent_repos = settings.AGENT_REPOSITORY_MAP.get(name, [])
        agent = spec['cls'](llm=llm, system_prompt=spec['prompt'], tools=spec['tools'])
        agents[name] = agent

        if 'route' in spec:
            specialists.append(
                RegistroEspecialista(
                    rota=spec['route'],
                    node_name=name,
                    agente=agent,
                    repositories=agent_repos
                )
            )

    graph = GraphBuilder(
        router_agent=agents['router_agent'],
        orchestrator_agent=agents['orchestrator_agent'],
        specialists=specialists,
        repositories=_REPOSITORIES_MAP,
        checkpointer=MemorySaver()
    ).build_graph()

    return Container(
        graph=graph,
        agentes=agents,
        factories=_FACTORIES_MAP,
        repositories=_REPOSITORIES_MAP,
        mongo_memory = mongo_memory
    )


class Container:
    def __init__(self, graph, agentes: Dict[str, BaseAgent], factories: Dict[str, Tuple[ConnectionFactory, Any]], repositories: Dict[str, Repository], mongo_memory: MongoMemory):
        self.graph = graph
        self.agentes = agentes
        self.factories = factories
        self.repositories = repositories
        self.mongo_memory = mongo_memory