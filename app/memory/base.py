from typing import Dict, List, Optional
from pydantic import BaseModel
from abc import ABC

from app.repository.base import Repository


class MemoriaCtx(BaseModel):
    resumo: Optional[str] = None
    mensagens_recentes: List[Dict] = []
    agentes_chamados: List[str] = []

class MemoryStore(ABC):
    def __init__(self, repository: Repository, janela_mensagens: int = 10):
        self.repository = repository
        self.janela_mensagens = janela_mensagens