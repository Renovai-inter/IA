from app.core.config import Settings
from app.graph.registro import RegistroEspecialista
from app.llms.factory import LLMFactory
from app.llms.embedding_service import EmbeddingService
from app.memory.base import MemoryStore
from app.memory.mongo_memory import MongoMemory
from app.memory.qdrant_memory import QdrantMemory
from app.llms.resumo_service import ResumoService
from app.prompts import (
    FAQ_PROMPT_COMPLETO,
    ROUTER_PROMPT_COMPLETO,
    MATERIAL_ESTOQUE_PROMPT_COMPLETO,
    ORQUESTRADOR_PROMPT_COMPLETO
)

from app.llms.gemini_provider import GeminiProvider
from app.llms.groq_provider import GroqProvider

from app.agents.base import BaseAgent
from app.agents.router_agent import RouterAgent
from app.agents.faq_agent import FaqAgent
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
from app.repository.qdrant.resumo_repository import ResumoRepository
from app.repository.qdrant.faq_chunks_repository import FaqChunksRepository
from app.rag.retriever import FaqRetriever
from app.tools.memory_tools import MemoriaToolkit
from app.tools.faq_chunks_tools import FaqToolkit
from app.tools.material_estoque_tools import MaterialEstoqueToolkit

from typing import Any, Dict, Tuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph


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

        'sessao_repository':               SessaoRepository(db=_FACTORIES_MAP.get('mongo_factory')[1], collection=settings.MONGO_COLLECTION_SESSAO),
        'resumo_repository':               ResumoRepository(db=_FACTORIES_MAP.get('qdrant_factory')[1], collection=settings.QDRANT_COLLECTION_MEMORIA),
        'faq_chunks_repository':           FaqChunksRepository(db=_FACTORIES_MAP.get('qdrant_factory')[1], collection=settings.QDRANT_COLLECTION_FAQ),
    }

    providers = {
        'GEMINI': GeminiProvider(settings.GEMINI_API_KEY),
        'GROQ':   GroqProvider(settings.GROQ_API_KEY),
    }
    llm_factory = LLMFactory(providers)

    # ResumoService (resume conversa) e EmbeddingService (vetoriza texto) são os
    # dois LLMService concretos: cada MemoryStore/retriever recebe o que
    # precisa de verdade, nunca os dois pelo mesmo nome de parâmetro.
    resumo_service = ResumoService(llm_factory, *settings.AGENT_LLM_MAP['resumo_agent'])
    embedding_service = EmbeddingService(settings.GEMINI_API_KEY)

    mongo_memory = MongoMemory(_REPOSITORIES_MAP['sessao_repository'], resumo_service)
    qdrant_memory = QdrantMemory(_REPOSITORIES_MAP['resumo_repository'], embedding_service)

    _MEMORY_STORE_MAP = {
        'mongo_memory': mongo_memory,
        'qdrant_memory': qdrant_memory,
    }

    faq_retriever = FaqRetriever(_REPOSITORIES_MAP['faq_chunks_repository'], embedding_service)

    memoria_toolkit = MemoriaToolkit(_MEMORY_STORE_MAP)
    faq_toolkit = FaqToolkit(faq_retriever)
    material_estoque_toolkit = MaterialEstoqueToolkit()

    _TOOLKITS_MAP = {
        'memoria_toolkit': memoria_toolkit,
        'faq_toolkit': faq_toolkit,
    }

    _AGENT_REGISTRY: dict[str, dict] = {
        'router_agent': {
            'cls': RouterAgent,
            'prompt': ROUTER_PROMPT_COMPLETO,
            'tools': memoria_toolkit.get_tools()
        },
        'faq_agent': {
            'route': 'rag_faq',
            'cls': FaqAgent,
            'prompt': FAQ_PROMPT_COMPLETO,
            'tools': faq_toolkit.get_tools()
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
        llm = llm_factory.get(*settings.AGENT_LLM_MAP[name])
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
        memory_stores=_MEMORY_STORE_MAP,
        toolkits=_TOOLKITS_MAP
    )


class Container:
    def __init__(
        self,
        graph: CompiledStateGraph,
        agentes: Dict[str, BaseAgent],
        factories: Dict[str, Tuple[ConnectionFactory, Any]],
        repositories: Dict[str, Repository],
        memory_stores: Dict[str, MemoryStore],
        toolkits: Dict[str, Any]
    ):
        self.graph = graph
        self.agentes = agentes
        self.factories = factories
        self.repositories = repositories
        self.memory_stores = memory_stores
        self.toolkits = toolkits
