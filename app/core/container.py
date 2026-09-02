from app.core.config import Settings
from app.llms.factory import LLMFactory
from app.prompts import (
    ROUTER_PROMPT,
    MATERIAL_ESTOQUE_PROMPT
)

from app.llms.gemini_provider import GeminiProvider
from app.llms.groq_provider import GroqProvider

from app.agents.base import BaseAgent
from app.agents.router_agent import RouterAgent
from app.agents.material_estoque_agent import MaterialEstoqueAgent

from app.graph.builder import GraphBuilder
from app.repository.postgresql.db import build_postgres_pool

from app.repository.base import Repository
from app.repository.postgresql.perfil_repository import PerfilRepository
from app.repository.postgresql.material_repository import MaterialRepository

from typing import Dict
from langgraph.checkpoint.memory import MemorySaver

_AGENT_REGISTRY: dict[str, dict] = {
    'router_agent': {'cls': RouterAgent, 'prompt': ROUTER_PROMPT, 'tools': []},
    'estoaue_material_agent': {'cls': MaterialEstoqueAgent, 'prompt': MATERIAL_ESTOQUE_PROMPT, 'tools': []}
}

def build_container(settings: Settings) -> Container:
    pg_pool = build_postgres_pool(settings.DATABASE_URL)

    _REPOSITORIES_MAP = {
        'perfil_repository':   PerfilRepository(db=pg_pool),
        'material_repository': MaterialRepository(db=pg_pool)
    }

    providers = {
        'GEMINI': GeminiProvider(settings.GEMINI_API_KEY),
        'GROQ':   GroqProvider(settings.GROQ_API_KEY),
    }
    factory = LLMFactory(providers)

    agents = {}
    for name, spec in _AGENT_REGISTRY.items():
        provider_name, tier = settings.AGENT_LLM_MAP[name]
        llm = factory.get(provider_name, tier)
        agents[name] = spec['cls'](llm=llm, system_prompt=spec['prompt'], tools=spec['tools'])

    graph = GraphBuilder(agents, _REPOSITORIES_MAP, MemorySaver()).build_graph()

    return Container(graph=graph, agentes=agents, pg_pool=pg_pool, repositories=_REPOSITORIES_MAP)


class Container:
    def __init__(self, graph, agentes: Dict[str, BaseAgent], pg_pool, repositories: Dict[str, Repository]):
        self.graph = graph
        self.agentes = agentes
        self.pg_pool = pg_pool
        self.repositories = repositories