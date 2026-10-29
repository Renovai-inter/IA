from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
)
from uuid import UUID
import operator
from langgraph.graph import MessagesState

from app.repository.postgresql.perfil_repository import PerfilContext, _to_uuid


@dataclass(slots=True)
class EspecialistaOutput:
    conteudo: str
    fonte: Optional[str] = None


RESET_OUTPUTS = '__reset__'
RESET_AGENTES = '__reset__'

def merge_especialista_outputs(
    existente: Dict[str, EspecialistaOutput],
    novo: Dict[str, EspecialistaOutput],
) -> Dict[str, EspecialistaOutput]:
    if novo and RESET_OUTPUTS in novo:
        return {}
    return {**(existente or {}), **(novo or {})}


def merge_agentes_chamados(
    existente: list[str],
    novo: list[str],
) -> list[str]:
    if novo and RESET_AGENTES in novo:
        return [item for item in novo if item != RESET_AGENTES]
    return (existente or []) + (novo or [])


class GraphState(MessagesState):
    cooperativa_id:       Optional[UUID]
    empresa_id:           Optional[UUID]
    perfil_ctx:           Optional[PerfilContext]
    proximos_agentes:     Optional[List[str]]
    agentes_chamados:     Annotated[list[str], merge_agentes_chamados]
    especialista_outputs: Annotated[dict[str, EspecialistaOutput], merge_especialista_outputs]
    chunk_contexto:       Optional[List[str]]
    passou_guardrail:     Optional[bool]
    motivo_guardrail:     Optional[str]
    passou_juiz:          Optional[bool]
    veredito_juiz:        Optional[str]
    score_juiz:           Optional[float]
    motivo_juiz:          Optional[str]
    frase_original:       Optional[str]


def resolver_contexto_dominio(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[UUID], Optional[UUID], str]:
    """Extrai `(cooperativa_id, empresa_id, tipo_fluxo)` diretamente do GraphState
    ou do RunnableConfig, mantendo fallback para `perfil_ctx` em testes legados."""
    cfg = ((config or {}).get('configurable') or {}) if isinstance(config, dict) else {}

    coop_raw = state.get('cooperativa_id') or cfg.get('cooperativa_id')
    emp_raw = state.get('empresa_id') or cfg.get('empresa_id')

    perfil_ctx = state.get('perfil_ctx')
    if coop_raw is None and perfil_ctx is not None:
        coop_raw = getattr(perfil_ctx, 'cooperativa_id', None)
    if emp_raw is None and perfil_ctx is not None:
        emp_raw = getattr(perfil_ctx, 'empresa_id', None)

    coop_uuid = _to_uuid(coop_raw)
    emp_uuid = _to_uuid(emp_raw)

    if coop_uuid is not None:
        return coop_uuid, None, 'COOPERATIVA'
    if emp_uuid is not None:
        return None, emp_uuid, 'EMPRESA'
    if perfil_ctx is not None and getattr(perfil_ctx, 'tipo', None):
        return None, None, perfil_ctx.tipo
    return None, None, 'DESCONHECIDO'
