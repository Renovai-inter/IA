from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod

from app.llms.base import LLMService
from app.repository.base import Repository


class MemoriaCtx(BaseModel):
    doc_id: str
    usuario_id: Optional[str]
    perfil_id: str
    sessao_id: str
    resumo: Optional[str] = None
    data_inicio: Optional[datetime] = None
    mensagens_recentes: List[Dict] = []
    agentes_chamados: List[str] = []

class MemoryStore(ABC):
    def __init__(self, repository: Repository, llm_service: LLMService, janela: int):
        self.repository = repository
        self.llm_service = llm_service
        self.janela = janela

    @abstractmethod
    def recuperar_historico(self, user_id: Optional[str], perfil_id: str, busca: Optional[str] = None) -> List[MemoriaCtx]:
        raise NotImplementedError