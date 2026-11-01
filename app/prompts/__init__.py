"""Pacote de Prompts Modularizados do Renovaí.

Reexporta todos os prompts compilados e templates para compatibilidade
retroativa com container.py, agentes e suites de teste.
"""
from __future__ import annotations

from app.prompts.base import (
    CONTEXTO_TEMPORAL,
    CONTRATO_JSON_ESPECIALISTA,
    PERSONA_SISTEMA,
    PERSONA_SISTEMA_NIMO,
    SHOTS_CUT,
    SHOTS_OPEN,
    obter_contexto_temporal,
)
from app.prompts.coleta import (
    COLETA_PROMPT,
    COLETA_PROMPT_COMPLETO,
)
from app.prompts.faq import (
    FAQ_PROMPT,
    FAQ_PROMPT_COMPLETO,
)
from app.prompts.financeiro import (
    FINANCEIRO_PROMPT,
    FINANCEIRO_PROMPT_COMPLETO,
)
from app.prompts.guardrails import (
    GUARDRAIL_ENTRADA_PROMPT,
    GUARDRAIL_ENTRADA_PROMPT_COMPLETO,
    GUARDRAIL_SAIDA_PROMPT,
    GUARDRAIL_SAIDA_PROMPT_COMPLETO,
)
from app.prompts.judge import (
    REGRAS_AVALIACAO,
    montar_prompt_juiz,
)
from app.prompts.material_estoque import (
    MATERIAL_ESTOQUE_PROMPT,
    MATERIAL_ESTOQUE_PROMPT_COMPLETO,
)
from app.prompts.orchestrator import (
    ORQUESTRADOR_PROMPT,
    ORQUESTRADOR_PROMPT_COMPLETO,
)
from app.prompts.recomendacao import (
    RECOMENDACAO_PROMPT,
    RECOMENDACAO_PROMPT_COMPLETO,
)
from app.prompts.router import (
    ROUTER_PROMPT,
    ROUTER_PROMPT_COMPLETO,
)

__all__ = [
    'PERSONA_SISTEMA',
    'PERSONA_SISTEMA_NIMO',
    'CONTEXTO_TEMPORAL',
    'obter_contexto_temporal',
    'CONTRATO_JSON_ESPECIALISTA',
    'SHOTS_OPEN',
    'SHOTS_CUT',
    'ROUTER_PROMPT',
    'ROUTER_PROMPT_COMPLETO',
    'MATERIAL_ESTOQUE_PROMPT',
    'MATERIAL_ESTOQUE_PROMPT_COMPLETO',
    'COLETA_PROMPT',
    'COLETA_PROMPT_COMPLETO',
    'FINANCEIRO_PROMPT',
    'FINANCEIRO_PROMPT_COMPLETO',
    'RECOMENDACAO_PROMPT',
    'RECOMENDACAO_PROMPT_COMPLETO',
    'FAQ_PROMPT',
    'FAQ_PROMPT_COMPLETO',
    'ORQUESTRADOR_PROMPT',
    'ORQUESTRADOR_PROMPT_COMPLETO',
    'GUARDRAIL_ENTRADA_PROMPT',
    'GUARDRAIL_ENTRADA_PROMPT_COMPLETO',
    'GUARDRAIL_SAIDA_PROMPT',
    'GUARDRAIL_SAIDA_PROMPT_COMPLETO',
    'REGRAS_AVALIACAO',
    'montar_prompt_juiz',
]
