"""Réplica do build_container (app/core/container.py) para os testes e2e,
com uma única diferença de propósito: a LLMFactory usa só OllamaProvider
(tiers HIGH/MEDIUM/LOW locais) no lugar de Gemini/Groq. Postgres, Mongo e
Qdrant são os REAIS do seu .env — não duplicamos infra pra isso, a pedido.

Por que uma cópia, e não um parâmetro em app/core/container.py:
`build_container` fica 100% intocado, então nada aqui pode quebrar a
wiring de produção (nem afetar o comportamento de app/main.py). O custo é
que, se `build_container` mudar (novo agente, novo repository, nova
tool), esta cópia pode ficar desatualizada — é a outra opção que te
propus (adicionar um `providers=` opcional em `build_container`) se você
preferir trocar por isso depois.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.agents.base import BaseAgent
from app.agents.material_estoque_agent import MaterialEstoqueAgent
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.router_agent import RouterAgent
from app.core.config import Settings
from app.core.container import Container
from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista
from app.llms.factory import LLMFactory
from app.llms.ollama_provider import OllamaProvider
from app.memory.mongo_memory import MongoMemory
from app.memory.qdrant_memory import QdrantMemory
from app.memory.resumo_service import ResumoService
from app.prompts import (
    MATERIAL_ESTOQUE_PROMPT_COMPLETO,
    ORQUESTRADOR_PROMPT_COMPLETO,
    ROUTER_PROMPT_COMPLETO,
)
from app.repository.mongodb.db import MongoConnectionFactory
from app.repository.mongodb.sessao_repository import SessaoRepository
from app.repository.postgresql.db import PostgresConnectionFactory
from app.repository.postgresql.estoque_repository import EstoqueRepository
from app.repository.postgresql.material_repository import MaterialRepository
from app.repository.postgresql.movimentacao_estoque_repository import (
    MovimentacaoEstoqueRepository,
)
from app.repository.postgresql.perfil_repository import PerfilRepository
from app.repository.qdrant.db import QdrantConnectionFactory
from app.repository.qdrant.resumo_repository import ResumoRepository
from app.tools.material_estoque_tools import MaterialEstoqueToolkit
from app.tools.memory_tools import MemoriaToolkit

# Espelha Settings.AGENT_LLM_MAP, só trocando o provider por 'OLLAMA' — os
# mesmos tiers (HIGH/MEDIUM/LOW) existem nos três providers.
_OLLAMA_AGENT_LLM_MAP: dict[str, tuple[str, str]] = {
    "router_agent": ("OLLAMA", "LOW"),
    "material_estoque_agent": ("OLLAMA", "HIGH"),
    "orchestrator_agent": ("OLLAMA", "LOW"),
    "resumo_agent": ("OLLAMA", "LOW"),
}


def build_e2e_container(settings: Settings) -> Container:
    pg_factory = PostgresConnectionFactory(dsn=settings.DATABASE_URL)
    mongo_factory = MongoConnectionFactory(dsn=settings.MONGODB_URI, db_name="mongo_dbrenovai")
    qdrant_factory = QdrantConnectionFactory(dsn=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    _FACTORIES_MAP = {
        "pg_factory": (pg_factory, pg_factory.connect()),
        "mongo_factory": (mongo_factory, mongo_factory.connect()),
        "qdrant_factory": (qdrant_factory, qdrant_factory.connect()),
    }

    _REPOSITORIES_MAP = {
        "perfil_repository": PerfilRepository(db=_FACTORIES_MAP["pg_factory"][1]),
        "material_repository": MaterialRepository(db=_FACTORIES_MAP["pg_factory"][1]),
        "estoque_repository": EstoqueRepository(db=_FACTORIES_MAP["pg_factory"][1]),
        "movimentacao_estoque_repository": MovimentacaoEstoqueRepository(db=_FACTORIES_MAP["pg_factory"][1]),
        "sessao_repository": SessaoRepository(db=_FACTORIES_MAP["mongo_factory"][1]),
        "resumo_repository": ResumoRepository(db=_FACTORIES_MAP["qdrant_factory"][1]),
    }

    providers = {"OLLAMA": OllamaProvider(base_url=settings.OLLAMA_BASE_URL)}
    llm_factory = LLMFactory(providers)

    # Atenção (herdada de app/core/container.py, não introduzida aqui):
    # ResumoService usa o MESMO llm (LOW) pra `_llm` (chat) e `_embeddings`
    # (.embed_query/.embed_documents). ChatOllama não implementa esses
    # métodos — assim como ChatGroq também não. Isso só quebra se algum
    # agente de fato chamar a tool `buscar_historico` com um termo de busca
    # (recuperar_historico via Qdrant); no fluxo comum de /chat isso não é
    # exercitado. Pré-existente, não é algo que os testes e2e corrigem.
    resumo_service = ResumoService(llm_factory, *_OLLAMA_AGENT_LLM_MAP["resumo_agent"])
    mongo_memory = MongoMemory(_REPOSITORIES_MAP["sessao_repository"], resumo_service)
    qdrant_memory = QdrantMemory(_REPOSITORIES_MAP["resumo_repository"], resumo_service)

    _MEMORY_STORE_MAP = {
        "mongo_memory": mongo_memory,
        "qdrant_memory": qdrant_memory,
    }

    memoria_toolkit = MemoriaToolkit(_MEMORY_STORE_MAP)
    material_estoque_toolkit = MaterialEstoqueToolkit()

    _AGENT_REGISTRY: dict[str, dict] = {
        "router_agent": {
            "cls": RouterAgent,
            "prompt": ROUTER_PROMPT_COMPLETO,
            "tools": memoria_toolkit.get_tools(),
        },
        "material_estoque_agent": {
            "route": "estoque",
            "cls": MaterialEstoqueAgent,
            "prompt": MATERIAL_ESTOQUE_PROMPT_COMPLETO,
            "tools": memoria_toolkit.get_tools() + material_estoque_toolkit.get_tools(),
        },
        "orchestrator_agent": {
            "cls": OrchestratorAgent,
            "prompt": ORQUESTRADOR_PROMPT_COMPLETO,
            "tools": [],
        },
    }

    specialists: list[RegistroEspecialista] = []
    agents: dict[str, BaseAgent] = {}
    for name, spec in _AGENT_REGISTRY.items():
        llm = llm_factory.get(*_OLLAMA_AGENT_LLM_MAP[name])
        agent_repos = settings.AGENT_REPOSITORY_MAP.get(name, [])
        agent = spec["cls"](llm=llm, system_prompt=spec["prompt"], tools=spec["tools"])
        agents[name] = agent

        if "route" in spec:
            specialists.append(
                RegistroEspecialista(
                    rota=spec["route"],
                    node_name=name,
                    agente=agent,
                    repositories=agent_repos,
                )
            )

    graph = GraphBuilder(
        router_agent=agents["router_agent"],
        orchestrator_agent=agents["orchestrator_agent"],
        specialists=specialists,
        repositories=_REPOSITORIES_MAP,
        checkpointer=MemorySaver(),
    ).build_graph()

    return Container(
        graph=graph,
        agentes=agents,
        factories=_FACTORIES_MAP,
        repositories=_REPOSITORIES_MAP,
        memory_stores=_MEMORY_STORE_MAP,
    )
