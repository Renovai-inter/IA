from typing import Sequence

from pydantic import BaseModel, ConfigDict

from app.agents.base import BaseAgent


class RegistroEspecialista(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    rota: str
    node_name: str
    agente: BaseAgent
    repositories: Sequence[str] = ()