from app.core.config import Settings
from app.llms.factory import LLMFactory
from app.prompts import ROUTER_PROMPT

from app.llms.gemini_provider import GeminiProvider
from app.llms.groq_provider import GroqProvider

from app.agents.router_agent import RouterAgent

class Container:
    def __init__(self, settings: Settings):
        self.settings = settings

        self._AGENT_REGISTRY: dict[str, dict] = {
            "router_agent": {
                "cls": RouterAgent,
                "prompt": ROUTER_PROMPT,
                "tools": [],
            },
        }

    def build_container(self):
        providers = {
            'GEMINI': GeminiProvider(self.settings.GEMINI_API_KEY),
            'GROQ' : GroqProvider(self.settings.GROQ_API_KEY),
        }
        factory = LLMFactory(providers)

        agents = {}
        for name, spec in self._AGENT_REGISTRY.items():
            provider_name, tier = self.settings.AGENT_LLM_MAP[name]
            llm = factory.get(provider_name, tier)
            agents[name] = spec["cls"](
                llm=llm,
                system_prompt=spec["prompt"],
                tools=spec["tools"],
            )

        # falta implementar os grafos e o StateGraph para retornar uma instância de Container(graph, agents)
        pass