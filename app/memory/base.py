from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod

from app.memory.resumo_service import ResumoService
from app.repository.base import Repository


class MemoriaCtx(BaseModel):
    doc_id: str
    resumo: Optional[str] = None
    data_inicio: Optional[datetime] = None
    mensagens_recentes: List[Dict] = []
    agentes_chamados: List[str] = []

class MemoryStore(ABC):
    def __init__(self, repository: Repository, resumo_service: ResumoService, janela: int):
        self.repository = repository
        self.resumo_service = resumo_service
        self.janela = janela

    @abstractmethod
    def recuperar_historico(self, user_id: Optional[str], perfil_id: str, busca: Optional[str] = None) -> List[MemoriaCtx]:
        raise NotImplementedError