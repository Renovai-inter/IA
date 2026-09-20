from typing import List

from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str
    resumo: str | None = None
    agentes_chamados: List[str] = []