from app.llms.base import LLMProvider

from langchain_core.language_models import BaseChatModel

class LLMFactory:
    def __init__(self, providers: dict[str, LLMProvider]):
        self._providers = providers

    def get(self, provider_name: str, tier: str, **overrides) -> BaseChatModel:
        primary_llm = self._providers[provider_name].get_llm(tier, **overrides)
        fallbacks = [
            llm.get_llm(tier, **overrides)
            for prov, llm in self._providers.items()
            if prov != provider_name
        ]
        fallbacks = list(filter(lambda x: x is not None, fallbacks))

        if fallbacks:
            return primary_llm.with_fallbacks(fallbacks)
            
        return primary_llm