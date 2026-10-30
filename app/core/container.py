from app.a2a.client import A2AClient
from app.a2a.server import A2ACooperativaServer, A2AEmpresaServer
from app.core.config import Settings
from app.core.observabilidade import Observabilidade
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
    RECOMENDACAO_PROMPT_COMPLETO,
    COLETA_PROMPT_COMPLETO,
    FINANCEIRO_PROMPT_COMPLETO,
    ORQUESTRADOR_PROMPT_COMPLETO,
    GUARDRAIL_ENTRADA_PROMPT_COMPLETO,
    GUARDRAIL_SAIDA_PROMPT_COMPLETO,
)

from app.llms.gemini_provider import GeminiProvider
from app.llms.groq_provider import GroqProvider

from app.agents.base import BaseAgent
from app.agents.router_agent import RouterAgent
from app.agents.faq_agent import FaqAgent
from app.agents.material_estoque_agent import MaterialEstoqueAgent
from app.agents.coleta_agent import ColetaAgent
from app.agents.financeiro_agent import FinanceiroAgent
from app.agents.recomendacao_agent import RecomendacaoAgent
from app.agents.logistica_agent import LogisticaAgent
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.guardrail_agent import GuardrailEntradaAgent, GuardrailSaidaAgent
from app.agents.judge_agent import JudgeAgent

from app.graph.builder import GraphBuilder
from app.repository.mongodb.db import MongoConnectionFactory
from app.repository.mongodb.sessao_repository import SessaoRepository
from app.repository.neo4j.db import Neo4jConnectionFactory
from app.repository.neo4j.recomendacao_repository import RecomendacaoRepository
from app.repository.neo4j.rota_repository import RotaRepository
from app.ml.route_model import ModeloRotaSupervisionadoSP
from app.repository.postgresql.db import PostgresConnectionFactory

from app.repository.base import ConnectionFactory, Repository
from app.repository.postgresql.perfil_repository import PerfilRepository
from app.repository.postgresql.material_repository import MaterialRepository
from app.repository.postgresql.estoque_repository import EstoqueRepository
from app.repository.postgresql.movimentacao_estoque_repository import MovimentacaoEstoqueRepository
from app.repository.postgresql.coleta_repository import ColetaRepository
from app.repository.postgresql.triagem_repository import TriagemRepository
from app.repository.postgresql.funcionario_repository import FuncionarioRepository
from app.repository.postgresql.equipe_repository import EquipeRepository
from app.repository.postgresql.rateio_repository import RateioRepository
from app.repository.postgresql.rateio_funcionario_repository import RateioFuncionarioRepository
from app.repository.postgresql.despesa_repository import DespesaRepository

from app.repository.qdrant.db import QdrantConnectionFactory
from app.repository.qdrant.resumo_repository import ResumoRepository
from app.repository.qdrant.faq_chunks_repository import FaqChunksRepository
from app.repository.redis import RedisConnectionFactory, RedisRealtimeRepository
from app.rag.retriever import FaqRetriever
from app.tools.memory_tools import MemoriaToolkit
from app.tools.faq_chunks_tools import FaqToolkit
from app.tools.material_estoque_tools import MaterialEstoqueToolkit
from app.tools.recomendacao_tools import RecomendacaoToolkit
from app.tools.coleta_tools import ColetaToolkit
from app.tools.financeiro_tools import FinanceiroToolkit

from typing import Any, Dict, Tuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph


def build_container(
    settings: Settings,
    providers: Dict[str, Any] | None = None,
    agent_llm_map: Dict[str, Tuple[str, str]] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> 'Container':
    """
    Construção do contêiner com:
    - Grafo Principal Unificado (Host Graph): `guardrail_entrada`, `router_agent`, `faq_agent`, `orchestrator_agent`, `guardrail_saida`, `judge`
    - Microsserviço A2A Cooperativa (`/a2a/cooperativa`): subgrafo `estoque`, `coleta`, `financeiro`, `logistica`
    - Microsserviço A2A Empresa (`/a2a/empresa`): subgrafo `pedidos` (`recomendacao`)
    """
    checkpointer = checkpointer or MemorySaver()

    pg_factory = PostgresConnectionFactory(dsn=settings.DATABASE_URL)
    mongo_factory = MongoConnectionFactory(dsn=settings.MONGODB_URI, db_name=settings.MONGO_DB_NAME)
    qdrant_factory = QdrantConnectionFactory(dsn=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    neo4j_factory = Neo4jConnectionFactory(
        dsn=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
    )
    redis_factory = RedisConnectionFactory(dsn=getattr(settings, 'REDIS_URL', ''))

    _FACTORIES_MAP = {
        'pg_factory':     (pg_factory,     pg_factory.connect()),
        'mongo_factory':  (mongo_factory,  mongo_factory.connect()),
        'qdrant_factory': (qdrant_factory, qdrant_factory.connect()),
        'neo4j_factory':  (neo4j_factory,  neo4j_factory.connect()),
        'redis_factory':  (redis_factory,  redis_factory.connect()),
    }

    _REPOSITORIES_MAP = {
        'perfil_repository':               PerfilRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'material_repository':             MaterialRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'estoque_repository':              EstoqueRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'movimentacao_estoque_repository': MovimentacaoEstoqueRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'coleta_repository':               ColetaRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'triagem_repository':              TriagemRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'funcionario_repository':          FuncionarioRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'equipe_repository':               EquipeRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'rateio_repository':               RateioRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'rateio_funcionario_repository':   RateioFuncionarioRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),
        'despesa_repository':              DespesaRepository(db=_FACTORIES_MAP.get('pg_factory')[1]),

        'sessao_repository':       SessaoRepository(
            db=_FACTORIES_MAP.get('mongo_factory')[1],
            collection=settings.MONGO_COLLECTION_SESSAO
        ),

        'resumo_repository':       ResumoRepository(
            db=_FACTORIES_MAP.get('qdrant_factory')[1],
            collection=settings.QDRANT_COLLECTION_MEMORIA
        ),

        'faq_chunks_repository':   FaqChunksRepository(
            db=_FACTORIES_MAP.get('qdrant_factory')[1],
            collection=settings.QDRANT_COLLECTION_FAQ
        ),

        'normas_repository':       FaqChunksRepository(
            db=_FACTORIES_MAP.get('qdrant_factory')[1],
            collection=settings.QDRANT_COLLECTION_NORMAS
        ),

        'recomendacao_repository':  RecomendacaoRepository(
            db=_FACTORIES_MAP.get('neo4j_factory')[1],
            database=getattr(settings, 'NEO4J_DATABASE', 'neo4j'),
        ),

        'rota_repository':          RotaRepository(
            db=_FACTORIES_MAP.get('neo4j_factory')[1],
            database=getattr(settings, 'NEO4J_DATABASE', 'neo4j'),
        ),

        'redis_realtime_repository': RedisRealtimeRepository(
            db=_FACTORIES_MAP.get('redis_factory')[1],
            max_requests=getattr(settings, 'RATE_LIMIT_MAX_REQUESTS', 15),
            window_seconds=getattr(settings, 'RATE_LIMIT_WINDOW_SECONDS', 60),
        ),
    }

    if providers is None:
        providers = {
            'GEMINI': GeminiProvider(settings.GEMINI_API_KEY),
            'GROQ':   GroqProvider(settings.GROQ_API_KEY),
        }

    observabilidade = Observabilidade()
    observabilidade.acoplar_provedores(providers)

    llm_factory = LLMFactory(providers)
    llm_map = agent_llm_map or settings.AGENT_LLM_MAP

    resumo_service = ResumoService(llm_factory, *llm_map['resumo_agent'])
    embedding_service = EmbeddingService(settings.HF_TOKEN, gemini_api_key=settings.GEMINI_API_KEY)

    mongo_memory = MongoMemory(_REPOSITORIES_MAP['sessao_repository'], resumo_service)
    qdrant_memory = QdrantMemory(_REPOSITORIES_MAP['resumo_repository'], embedding_service)

    _MEMORY_STORE_MAP = {
        'mongo_memory': mongo_memory,
        'qdrant_memory': qdrant_memory,
    }

    faq_retriever = FaqRetriever(
        repository=_REPOSITORIES_MAP['faq_chunks_repository'],
        embedding_service=embedding_service,
        normas_repository=_REPOSITORIES_MAP['normas_repository'],
    )

    memoria_toolkit = MemoriaToolkit(_MEMORY_STORE_MAP)
    faq_toolkit = FaqToolkit(faq_retriever)
    recomendacao_toolkit = RecomendacaoToolkit(_REPOSITORIES_MAP['recomendacao_repository'])

    material_estoque_toolkit = MaterialEstoqueToolkit()
    coleta_toolkit = ColetaToolkit(_REPOSITORIES_MAP)
    financeiro_toolkit = FinanceiroToolkit(_REPOSITORIES_MAP)

    _TOOLKITS_MAP = {
        'memoria_toolkit':          memoria_toolkit,
        'faq_toolkit':              faq_toolkit,
        'recomendacao_toolkit':     recomendacao_toolkit,
        'material_estoque_toolkit': material_estoque_toolkit,
        'coleta_toolkit':           coleta_toolkit,
        'financeiro_toolkit':       financeiro_toolkit,
    }

    modelo_rotas = ModeloRotaSupervisionadoSP()

    _AGENT_REGISTRY: dict[str, dict] = {
        'guardrail_entrada': {
            'cls': GuardrailEntradaAgent,
            'prompt': GUARDRAIL_ENTRADA_PROMPT_COMPLETO,
            'tools': []
        },
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
        'recomendacao_agent': {
            'route': 'pedidos',
            'cls': RecomendacaoAgent,
            'prompt': RECOMENDACAO_PROMPT_COMPLETO,
            'tools': memoria_toolkit.get_tools() + recomendacao_toolkit.get_tools()
        },
        'logistica_agent': {
            'route': 'logistica',
            'cls': LogisticaAgent,
            'prompt': '',
            'tools': [],
            'sem_llm': True,
            'kwargs': {
                'modelo_rotas': modelo_rotas,
                'rota_repository': _REPOSITORIES_MAP['rota_repository'],
            },
        },
        'coleta_agent': {
            'route': 'coleta',
            'cls': ColetaAgent,
            'prompt': COLETA_PROMPT_COMPLETO,
            'tools': memoria_toolkit.get_tools() + coleta_toolkit.get_tools()
        },
        'financeiro_agent': {
            'route': 'financeiro',
            'cls': FinanceiroAgent,
            'prompt': FINANCEIRO_PROMPT_COMPLETO,
            'tools': memoria_toolkit.get_tools() + financeiro_toolkit.get_tools()
        },
        'orchestrator_agent': {
            'cls': OrchestratorAgent,
            'prompt': ORQUESTRADOR_PROMPT_COMPLETO,
            'tools': []
        },
        'guardrail_saida': {
            'cls': GuardrailSaidaAgent,
            'prompt': GUARDRAIL_SAIDA_PROMPT_COMPLETO,
            'tools': [],
            'rotas_revisao': getattr(settings, 'GUARDRAIL_ROTAS_REVISAO', ('financeiro',)),
        },
        'judge': {
            'cls': JudgeAgent,
            'prompt': '',
            'tools': [],
            'sem_llm': True,
            'kwargs': {
                'api_key': getattr(settings, 'LAYA_API_KEY', ''),
                'api_url': getattr(settings, 'LAYA_API_URL', 'https://api.laya.studio/v1/systemone'),
                'model': getattr(settings, 'LAYA_MODEL', 'multilingual'),
            },
        },
    }

    specialists: list[RegistroEspecialista] = []
    agents = {}
    for name, spec in _AGENT_REGISTRY.items():
        if spec.get('sem_llm'):
            llm = None
        else:
            provider_tier = llm_map.get(name) or settings.AGENT_LLM_MAP.get(name, ('GEMINI', 'HIGH'))
            llm = llm_factory.get(*provider_tier)
        agent_repos = settings.AGENT_REPOSITORY_MAP.get(name, [])
        kwargs = dict(spec.get('kwargs') or {})
        if 'rotas_revisao' in spec:
            kwargs['rotas_revisao'] = spec['rotas_revisao']
        agent = spec['cls'](llm=llm, system_prompt=spec['prompt'], tools=spec['tools'], **kwargs)
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

    rotas_cooperativa = set(getattr(settings, 'ROTAS_COOPERATIVA', ('estoque', 'coleta', 'financeiro', 'logistica')))
    rotas_empresa = set(getattr(settings, 'ROTAS_EMPRESA', ('pedidos',)))

    specialists_cooperativa = [s for s in specialists if s.rota in rotas_cooperativa]
    specialists_empresa = [s for s in specialists if s.rota in rotas_empresa]

    a2a_cooperativa_server = A2ACooperativaServer(
        specialists=specialists_cooperativa,
        repositories=_REPOSITORIES_MAP,
    )
    a2a_empresa_server = A2AEmpresaServer(
        specialists=specialists_empresa,
        repositories=_REPOSITORIES_MAP,
    )
    a2a_client = A2AClient(
        cooperativa_url=(getattr(settings, 'COOPERATIVA_A2A_URL', '') or '').strip() or None,
        empresa_url=(getattr(settings, 'EMPRESA_A2A_URL', '') or '').strip() or None,
        cooperativa_server=a2a_cooperativa_server,
        empresa_server=a2a_empresa_server,
    )

    graph = GraphBuilder(
        router_agent=agents['router_agent'],
        orchestrator_agent=agents['orchestrator_agent'],
        specialists=specialists,
        repositories=_REPOSITORIES_MAP,
        checkpointer=checkpointer,
        guardrail_entrada=agents['guardrail_entrada'],
        guardrail_saida=agents['guardrail_saida'],
        a2a_client=a2a_client,
        judge_agent=agents['judge'],
    ).build_graph()

    return Container(
        graph=graph,
        agentes=agents,
        factories=_FACTORIES_MAP,
        repositories=_REPOSITORIES_MAP,
        memory_stores=_MEMORY_STORE_MAP,
        toolkits=_TOOLKITS_MAP,
        observabilidade=observabilidade,
        checkpointer=checkpointer,
        modelo_rotas=modelo_rotas,
        a2a_cooperativa_server=a2a_cooperativa_server,
        a2a_empresa_server=a2a_empresa_server,
        a2a_client=a2a_client,
        judge_agent=agents['judge'],
    )


class Container:
    def __init__(
        self,
        graph: CompiledStateGraph,
        agentes: Dict[str, BaseAgent],
        factories: Dict[str, Tuple[ConnectionFactory, Any]],
        repositories: Dict[str, Repository],
        memory_stores: Dict[str, MemoryStore],
        toolkits: Dict[str, Any],
        observabilidade: Observabilidade | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        modelo_rotas: ModeloRotaSupervisionadoSP | None = None,
        a2a_cooperativa_server: A2ACooperativaServer | None = None,
        a2a_empresa_server: A2AEmpresaServer | None = None,
        a2a_client: A2AClient | None = None,
        judge_agent: BaseAgent | None = None,
    ):
        self.graph = graph
        self.agentes = agentes
        self.factories = factories
        self.repositories = repositories
        self.memory_stores = memory_stores
        self.toolkits = toolkits
        self.observabilidade = observabilidade or Observabilidade()
        self.checkpointer = checkpointer
        self.modelo_rotas = modelo_rotas or ModeloRotaSupervisionadoSP()
        self.a2a_cooperativa_server = a2a_cooperativa_server or A2ACooperativaServer(repositories=repositories)
        self.a2a_empresa_server = a2a_empresa_server or A2AEmpresaServer(repositories=repositories)
        self.a2a_server = self.a2a_cooperativa_server
        self.a2a_client = a2a_client or A2AClient(
            cooperativa_server=self.a2a_cooperativa_server,
            empresa_server=self.a2a_empresa_server,
        )
        self.judge_agent = judge_agent or self.agentes.get('judge')
        self.redis_realtime: RedisRealtimeRepository = (
            self.repositories.get('redis_realtime_repository') or RedisRealtimeRepository()
        )
