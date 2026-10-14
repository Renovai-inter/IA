from dataclasses import dataclass
from typing import Sequence

from app.agents.base import BaseAgent


@dataclass(slots=True)
class RegistroEspecialista:
    rota: str
    node_name: str
    agente: BaseAgent
    repositories: Sequence[str] = ()