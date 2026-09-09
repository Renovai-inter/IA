from app.repository.base import Repository

from langchain_core.tools import StructuredTool
from abc import ABC, abstractmethod
from typing import Dict

class Toolkit(ABC):
    """
    Contrato abstrato para a criação de toolkits
    que podem ser chamados por agentes do sistema.
    """
    nome: str
    descricao: str
    repositories: Dict[str, Repository]

    @abstractmethod
    def get_tools(self) -> list[StructuredTool]:
        raise NotImplementedError