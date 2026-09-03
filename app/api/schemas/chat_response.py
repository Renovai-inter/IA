from pydantic import BaseModel, Field
from typing import Any

class ChatResponse(BaseModel):
    """O que a API devolve no POST /chat."""
    resposta:         Any
    agentes_chamados: list[str] = Field(default_factory=list) 