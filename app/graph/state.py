from app.repository.postgresql.perfil_repository import PerfilContext

from pydantic import BaseModel
from langgraph.graph import MessagesState
from typing import (
    Annotated,
    Optional,
    List,
    Literal,
    Any,
    Dict
)
import operator

class EspecialistaOutput(BaseModel):
    conteudo: Dict[str, Any]
    fonte: Optional[str] = None
    veredito_juiz: Literal["pendente", "aprovado", "reprovado"] = "pendente"
    feedback_juiz: Optional[str] = None
    tentativas: int = 0


def merge_especialista_outputs(
    existente: Dict[str, EspecialistaOutput],
    novo: Dict[str, EspecialistaOutput],
) -> Dict[str, EspecialistaOutput]:
    return {**existente, **novo}


class GraphState(MessagesState):
    perfil_ctx :          Optional[PerfilContext]
    proximo_agente :      Optional[str]
    agentes_chamados :    Annotated[list[str], operator.add]
    especialista_outputs: Annotated[dict[str, EspecialistaOutput], merge_especialista_outputs]
    chunk_contexto :      Optional[List[str]]
    passou_guardrail :    Optional[bool]
    motivo_guardrail :    Optional[str]