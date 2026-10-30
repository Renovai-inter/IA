from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.fallbacks import RunnableWithFallbacks

log = logging.getLogger('renovai.llm.circuit_breaker')


def is_rate_limit_or_overload(exc: BaseException) -> bool:
    """Detecta se uma exceção é decorrente de rate limit (429), cota excedida ou timeout."""
    texto = str(exc).lower()
    indicadores = (
        '429',
        'rate limit',
        'ratelimit',
        'too many requests',
        'quota exceeded',
        'resource_exhausted',
        'resourceexhausted',
        'overloaded',
        'capacity',
        'deadline_exceeded',
        'timed out',
        'timeout',
    )
    return any(ind in texto for ind in indicadores)


class CircuitBreaker:
    """Disjuntor inteligente (Circuit Breaker) por provedor de LLM (thread-safe).

    Evita que o sistema repita tentativas em um provedor com cota esgotada (429) ou instável,
    eliminando a penalidade de latência de 10s-20s em cada requisição subsequente.
    """

    def __init__(self, cooldown_seconds: float = 30.0):
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._aberto_ate: dict[str, float] = {}
        self._falhas_consecutivas: dict[str, int] = {}

    def esta_aberto(self, provedor: str) -> bool:
        with self._lock:
            ate = self._aberto_ate.get(provedor, 0.0)
            agora = time.time()
            if agora < ate:
                return True
            if ate > 0.0:
                # Período de cooldown expirou: permite tentativa de reabilitação (half-open)
                self._aberto_ate.pop(provedor, None)
                self._falhas_consecutivas[provedor] = 0
            return False

    def registrar_falha(self, provedor: str, exc: Optional[BaseException] = None) -> None:
        com_rate_limit = is_rate_limit_or_overload(exc) if exc else False
        with self._lock:
            if com_rate_limit:
                self._aberto_ate[provedor] = time.time() + self._cooldown
                log.warning(
                    f"[circuit_breaker] Rate limit/Quota detectado em '{provedor}'. "
                    f"Disjuntor ABERTO por {self._cooldown:.0f}s. Requisições desviarão em 0ms para contingência."
                )
                return

            falhas = self._falhas_consecutivas.get(provedor, 0) + 1
            self._falhas_consecutivas[provedor] = falhas
            if falhas >= 2:
                self._aberto_ate[provedor] = time.time() + self._cooldown
                log.warning(
                    f"[circuit_breaker] {falhas} falhas consecutivas em '{provedor}'. "
                    f"Disjuntor ABERTO por {self._cooldown:.0f}s."
                )

    def registrar_sucesso(self, provedor: str) -> None:
        with self._lock:
            self._falhas_consecutivas[provedor] = 0
            self._aberto_ate.pop(provedor, None)


# Instância global singleton do Circuit Breaker para os provedores de LLM
_circuito_global = CircuitBreaker(cooldown_seconds=30.0)


class SmartFallbackRunnable(RunnableWithFallbacks):
    """Extensão de RunnableWithFallbacks que desvia chamadas em 0ms quando o primário está aberto."""

    primary_provider: str = ''
    circuit_breaker: Any = None

    def _obter_candidatos(self) -> list[Runnable]:
        cb = self.circuit_breaker or _circuito_global
        if self.primary_provider and cb.esta_aberto(self.primary_provider):
            log.info(
                f"[circuit_breaker] Desviando chamada de '{self.primary_provider}' diretamente para fallback (0ms de espera)."
            )
            return list(self.fallbacks) + [self.runnable]
        return [self.runnable] + list(self.fallbacks)

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        candidatos = self._obter_candidatos()
        ultimo_erro = None
        cb = self.circuit_breaker or _circuito_global
        for i, r in enumerate(candidatos):
            try:
                res = r.invoke(input, config=config, **kwargs)
                if i == 0 and self.primary_provider and not cb.esta_aberto(self.primary_provider):
                    cb.registrar_sucesso(self.primary_provider)
                return res
            except Exception as exc:
                ultimo_erro = exc
                if i == 0 and self.primary_provider:
                    cb.registrar_falha(self.primary_provider, exc)
                continue
        raise ultimo_erro

    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        candidatos = self._obter_candidatos()
        ultimo_erro = None
        cb = self.circuit_breaker or _circuito_global
        for i, r in enumerate(candidatos):
            try:
                res = await r.ainvoke(input, config=config, **kwargs)
                if i == 0 and self.primary_provider and not cb.esta_aberto(self.primary_provider):
                    cb.registrar_sucesso(self.primary_provider)
                return res
            except Exception as exc:
                ultimo_erro = exc
                if i == 0 and self.primary_provider:
                    cb.registrar_falha(self.primary_provider, exc)
                continue
        raise ultimo_erro
