"""
Rota de telemetria e monitoramento do Renovai IA.

Disponibiliza o endpoint GET /monitor com métricas de SRE (SLIs/SLOs),
métricas de modelos LLM (tokens, custos, fallbacks) e logs unificados.
"""

from fastapi import APIRouter, Request

router = APIRouter(tags=["monitor"])


@router.get("/monitor")
def painel(request: Request) -> dict:
    """Números e telemetria do processo atual do Renovai."""
    obs = request.app.state.container.observabilidade
    return obs.montar_painel()
