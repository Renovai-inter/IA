from app.repository.base import Repository
from app.core.observabilidade import timed as obs_timed

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

    @staticmethod
    def timed(categoria: str, nome: str):
        return obs_timed(nome=nome, categoria=categoria)

    @abstractmethod
    def get_tools(self) -> list[StructuredTool]:
        raise NotImplementedError