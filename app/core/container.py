from app.core.config import Settings
from app.llms.factory import LLMFactory
from app.prompts import ROUTER_PROMPT

from app.llms.gemini_provider import GeminiProvider
from app.llms.groq_provider import GroqProvider

from app.agents.base import BaseAgent
from app.agents.router_agent import RouterAgent

from app.graph.builder import GraphBuilder

from typing import Dict
from langgraph.checkpoint.memory import MemorySaver

_AGENT_REGISTRY: dict[str, dict] = {
    "router_agent": {"cls": RouterAgent, "prompt": ROUTER_PROMPT, "tools": []},
}

def build_container(settings: Settings) -> Container:
    providers = {
        'GEMINI': GeminiProvider(settings.GEMINI_API_KEY),
        'GROQ':   GroqProvider(settings.GROQ_API_KEY),
    }
    factory = LLMFactory(providers)

    agents = {}
    for name, spec in _AGENT_REGISTRY.items():
        provider_name, tier = settings.AGENT_LLM_MAP[name]
        llm = factory.get(provider_name, tier)
        agents[name] = spec["cls"](llm=llm, system_prompt=spec["prompt"], tools=spec["tools"])

    graph = GraphBuilder(agents, MemorySaver()).build_graph()
    return Container(graph=graph, agentes=agents)


class Container:
    def __init__(self, graph, agentes: Dict[str, BaseAgent]):
        self.graph = graph
        self.agentes = agentes