from dataclasses import dataclass
from typing import (
    Annotated,
    Dict,
    List,
    Literal,
    Optional,
)
import operator
from langgraph.graph import MessagesState

from app.repository.postgresql.perfil_repository import PerfilContext


@dataclass(slots=True)
class EspecialistaOutput:
    conteudo: str
    fonte: Optional[str] = None
    veredito_juiz: Literal["pendente", "aprovado", "reprovado"] = "pendente"
    feedback_juiz: Optional[str] = None
    tentativas: int = 0


RESET_OUTPUTS = '__reset__'

def merge_especialista_outputs(
    existente: Dict[str, EspecialistaOutput],
    novo: Dict[str, EspecialistaOutput],
) -> Dict[str, EspecialistaOutput]:
    if novo and RESET_OUTPUTS in novo:
        return {}
    return {**(existente or {}), **(novo or {})}


class GraphState(MessagesState):
    perfil_ctx :          Optional[PerfilContext]
    proximos_agentes :    Optional[List[str]]
    agentes_chamados :    Annotated[list[str], operator.add]
    especialista_outputs: Annotated[dict[str, EspecialistaOutput], merge_especialista_outputs]
    chunk_contexto :      Optional[List[str]]
    passou_guardrail :    Optional[bool]
    motivo_guardrail :    Optional[str]