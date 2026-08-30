from app.repository.postgresql.perfil_repository import PerfilContext

from langgraph.graph import MessagesState
from typing import (
    Annotated,
    Optional,
    List,
)
import operator

class GraphState(MessagesState):
    perfil_ctx       :  Optional[PerfilContext]
    agentes_chamados :  Annotated[list[str], operator.add]
    proximo_agente   :  Optional[str]
    chunk_contexto   :  Optional[List[str]]
    passou_guardrail :  Optional[bool]
    motivo_guardrail :  Optional[str]
    judge_veredict   :  Optional[str]
    judge_feedback   :  Optional[str]
    cont_tentativas  :  int