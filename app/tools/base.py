from app.repository.base import Repository

from langchain_core.tools import StructuredTool
from abc import ABC, abstractmethod
from typing import Dict

import inspect
import time as tm, functools, logging

import logging
log = logging.getLogger(__name__)


class Toolkit(ABC):
    """
    Contrato abstrato para a criação de toolkits
    que podem ser chamados por agentes do sistema.
    """
    nome: str
    descricao: str
    repositories: Dict[str, Repository]

    log = logging.getLogger(__name__)

    def timed(categoria, nome):
        log = logging.getLogger("latency")
        def deco(fn):
    
            if inspect.iscoroutinefunction(fn):
    
                @functools.wraps(fn)
                async def async_wrapper(*a, **kw):
                    t = tm.perf_counter()
    
                    try:
                        return await fn(*a, **kw)
                    finally:
                        log.warning(
                            "timing category=%s name=%s duration=%.2fs",
                            categoria,
                            nome,
                            tm.perf_counter() - t,
                        )
    
                return async_wrapper
    
            @functools.wraps(fn)
            def sync_wrapper(*a, **kw):
                t = tm.perf_counter()
    
                try:
                    return fn(*a, **kw)
                finally:
                    log.warning(
                        "timing category=%s name=%s duration=%.2fs",
                        categoria,
                        nome,
                        tm.perf_counter() - t,
                    )
    
            return sync_wrapper
    
        return deco

    @abstractmethod
    def get_tools(self) -> list[StructuredTool]:
        raise NotImplementedError