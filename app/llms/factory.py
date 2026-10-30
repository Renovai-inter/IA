from app.llms.base import LLMProvider
from app.llms.circuit_breaker import SmartFallbackRunnable, _circuito_global


class LLMFactory:
    def __init__(self, providers: dict[str, LLMProvider]):
        self._providers = providers

    def get(self, provider_name: str, tier: str, **overrides):
        primary_llm = self._providers[provider_name].get_llm(tier, **overrides)
        fallbacks = [
            llm.get_llm(tier, **overrides)
            for prov, llm in self._providers.items()
            if prov != provider_name
        ]
        fallbacks = [fb for fb in fallbacks if fb is not None]

        if fallbacks:
            return SmartFallbackRunnable(
                runnable=primary_llm,
                fallbacks=fallbacks,
                primary_provider=provider_name,
                circuit_breaker=_circuito_global,
            )

        return primary_llm
