"""Servidor MCP Standalone do Renovaí IA (Model Context Protocol — transporte stdio).

Compatível com Cursor (.cursor/mcp.json), Claude Desktop (claude_desktop_config.json),
VS Code (.vscode/mcp.json) e Antigravity.
Permite descoberta e invocação de ferramentas, recursos operacionais (PNRS, SRE)
e templates de prompt via linha de comando (stdin/stdout).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Annotated, Any, List, Optional
from uuid import UUID

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from app.core.config import Settings
from app.core.container import build_container

log = logging.getLogger("renovai.mcp_server")

mcp = MCPServer(
    name="renovai-mcp",
    version="1.0.0",
    instructions=(
        "Servidor de ferramentas e recursos contextuais do Renovaí IA para cooperativas de reciclagem e empresas. "
        "Permite consultar saldos e histórico de estoque de recicláveis (PET, Papelão, PEAD, Alumínio, etc.), "
        "agendamentos de coletas e triagens em esteira, rateios e finanças de cooperados, "
        "recomendações de sourcing de materiais para empresas, conformidade com a PNRS (Lei nº 12.305/2010) "
        "e telemetria SRE."
    ),
)

LEITURA = ToolAnnotations(readOnlyHint=True)
ESCRITA = ToolAnnotations(readOnlyHint=False, destructiveHint=False)

_container = None


def _obter_container():
    global _container
    if _container is None:
        settings = Settings()
        _container = build_container(settings)
    return _container


def _executar_tool(nome: str, argumentos: dict[str, Any], meta: Optional[dict[str, Any]] = None) -> Any:
    container = _obter_container()
    res = container.mcp_server.call_tool(nome, argumentos, meta=meta)
    if res.structuredContent is not None:
        return res.structuredContent
    if res.content and len(res.content) > 0:
        try:
            return json.loads(res.content[0].text)
        except Exception:
            return res.content[0].text
    return {"status": "ok"}


# ==============================================================================
# Ferramentas: Estoque e Materiais Recicláveis
# ==============================================================================
@mcp.tool(
    name="consultar_estoque",
    title="Consultar saldo em estoque",
    description="Consulta o saldo atual de materiais recicláveis em estoque na cooperativa (PET, Papelão, PEAD, etc.), com filtros de quantidade e preço.",
    annotations=LEITURA,
)
def consultar_estoque(
    material_nomes: Annotated[Optional[List[str]], Field(description="Nomes ou partes dos nomes dos materiais a filtrar (ex: ['PET', 'papelão']).")] = None,
    material_ids: Annotated[Optional[List[str]], Field(description="Lista de UUIDs de materiais específicos.")] = None,
    disponivel: Annotated[Optional[bool], Field(description="Filtrar por disponibilidade (True = apenas disponíveis).")] = None,
    quantidade_min_kg: Annotated[Optional[float], Field(description="Quantidade mínima em estoque em kg.")] = None,
    quantidade_max_kg: Annotated[Optional[float], Field(description="Quantidade máxima em estoque em kg.")] = None,
    incluir_sem_estoque: Annotated[bool, Field(description="Se True, inclui materiais com estoque zerado.")] = False,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa. Se omitido, usa a cooperativa ativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_estoque",
        {
            "material_nomes": material_nomes,
            "material_ids": material_ids,
            "disponivel": disponivel,
            "quantidade_min_kg": quantidade_min_kg,
            "quantidade_max_kg": quantidade_max_kg,
            "incluir_sem_estoque": incluir_sem_estoque,
        },
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


@mcp.tool(
    name="generalizar_estoque",
    title="Generalizar estoque por categoria",
    description="Agrupa e consolida os saldos de estoque por categoria-pai de material (ex: Plásticos, Papéis, Metais).",
    annotations=LEITURA,
)
def generalizar_estoque(
    disponivel: Annotated[Optional[bool], Field(description="Filtrar apenas disponíveis.")] = None,
    quantidade_min_kg: Annotated[Optional[float], Field(description="Quantidade mínima acumulada em kg.")] = None,
    incluir_sem_estoque: Annotated[bool, Field(description="Incluir categorias sem saldo.")] = False,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "generalizar_estoque",
        {
            "disponivel": disponivel,
            "quantidade_min_kg": quantidade_min_kg,
            "incluir_sem_estoque": incluir_sem_estoque,
        },
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


@mcp.tool(
    name="buscar_estoque_historico",
    title="Histórico de movimentações de estoque",
    description="Consulta o histórico temporal de entradas e saídas de materiais no estoque da cooperativa.",
    annotations=LEITURA,
)
def buscar_estoque_historico(
    material_nomes: Annotated[Optional[List[str]], Field(description="Nomes dos materiais.")] = None,
    tipo_movimentacao: Annotated[Optional[str], Field(description="Tipo: 'ENTRADA' (triagem) ou 'SAIDA' (venda/pedido).")] = None,
    data_inicio: Annotated[Optional[str], Field(description="Data inicial no formato ISO 8601.")] = None,
    data_fim: Annotated[Optional[str], Field(description="Data final no formato ISO 8601.")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "buscar_estoque_historico",
        {
            "material_nomes": material_nomes,
            "tipo_movimentacao": tipo_movimentacao,
            "data_inicio": data_inicio,
            "data_fim": data_fim,
        },
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


# ==============================================================================
# Ferramentas: Coleta e Triagem
# ==============================================================================
@mcp.tool(
    name="consultar_coletas",
    title="Consultar coletas e rotas",
    description="Consulta agendamentos de coleta de resíduos na cooperativa, filtrando por status (agendada, em andamento, concluída) e período.",
    annotations=LEITURA,
)
def consultar_coletas(
    status: Annotated[Optional[str], Field(description="Status da coleta: 'agendada', 'em_andamento', 'concluida'.")] = None,
    data_inicio: Annotated[Optional[str], Field(description="Data de início (ISO 8601).")] = None,
    data_fim: Annotated[Optional[str], Field(description="Data de fim (ISO 8601).")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_coletas",
        {
            "status": status,
            "data_inicio": data_inicio,
            "data_fim": data_fim,
        },
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


@mcp.tool(
    name="consultar_triagens",
    title="Consultar triagens em esteira",
    description="Consulta lotes triados e materiais separados pela cooperativa na esteira de triagem.",
    annotations=LEITURA,
)
def consultar_triagens(
    coleta_id: Annotated[Optional[str], Field(description="Filtrar por UUID de uma coleta específica.")] = None,
    data_inicio: Annotated[Optional[str], Field(description="Data de início.")] = None,
    data_fim: Annotated[Optional[str], Field(description="Data de fim.")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_triagens",
        {
            "coleta_id": coleta_id,
            "data_inicio": data_inicio,
            "data_fim": data_fim,
        },
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


@mcp.tool(
    name="consultar_equipes",
    title="Consultar equipes operacionais",
    description="Consulta equipes de campo e equipes de triagem cadastradas na cooperativa.",
    annotations=LEITURA,
)
def consultar_equipes(
    equipe_id: Annotated[Optional[str], Field(description="UUID de equipe específica.")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_equipes",
        {"equipe_id": equipe_id},
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


@mcp.tool(
    name="consultar_funcionarios",
    title="Consultar funcionários e cooperados",
    description="Consulta o quadro de cooperados e funcionários, com papéis e status de atividade.",
    annotations=LEITURA,
)
def consultar_funcionarios(
    papel: Annotated[Optional[str], Field(description="Papel: 'coletor', 'triador', 'motorista', 'administrativo'.")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_funcionarios",
        {"papel": papel},
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


# ==============================================================================
# Ferramentas: Financeiro e Rateio
# ==============================================================================
@mcp.tool(
    name="consultar_rateios",
    title="Consultar rateios financeiros",
    description="Consulta o fechamento de rateio mensal da cooperativa e valor líquido apurado por cota.",
    annotations=LEITURA,
)
def consultar_rateios(
    mes_ano: Annotated[Optional[str], Field(description="Mês de referência (ex: '2026-09' ou 'setembro/2026').")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_rateios",
        {"mes_ano": mes_ano},
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


@mcp.tool(
    name="consultar_despesas",
    title="Consultar despesas operacionais",
    description="Consulta custos e despesas operacionais da cooperativa por categoria (manutenção, combustível, água, luz, EPI).",
    annotations=LEITURA,
)
def consultar_despesas(
    categoria: Annotated[Optional[str], Field(description="Categoria de despesa.")] = None,
    mes_ano: Annotated[Optional[str], Field(description="Mês de referência (ex: '2026-09').")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_despesas",
        {"categoria": categoria, "mes_ano": mes_ano},
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


@mcp.tool(
    name="consultar_pagamentos_funcionarios",
    title="Consultar pagamentos individuais",
    description="Consulta valores pagos a cooperados ou funcionários referentes ao rateio de produção.",
    annotations=LEITURA,
)
def consultar_pagamentos_funcionarios(
    funcionario_id: Annotated[Optional[str], Field(description="UUID do cooperado/funcionário.")] = None,
    mes_ano: Annotated[Optional[str], Field(description="Mês de referência.")] = None,
    cooperativa_id: Annotated[Optional[str], Field(description="UUID da cooperativa.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "consultar_pagamentos_funcionarios",
        {"funcionario_id": funcionario_id, "mes_ano": mes_ano},
        meta={"cooperativa_id": cooperativa_id} if cooperativa_id else None,
    )


# ==============================================================================
# Ferramentas: Recomendação de Cooperativas (Empresas)
# ==============================================================================
@mcp.tool(
    name="recomendar_cooperativas",
    title="Recomendar cooperativas para empresas",
    description="Recomenda cooperativas fornecedoras de recicláveis com base no volume demandado, tipos de material e proximidade geográfica.",
    annotations=LEITURA,
)
def recomendar_cooperativas(
    tipo_material: Annotated[Optional[str], Field(description="Tipo de resíduo demandado (ex: PET, Papelão, Alumínio).")] = None,
    quantidade_minima_kg: Annotated[Optional[float], Field(description="Volume mínimo demandado em kg.")] = None,
    raio_km: Annotated[Optional[float], Field(description="Raio máximo de distância em km.")] = None,
    empresa_id: Annotated[Optional[str], Field(description="UUID da empresa compradora.")] = None,
) -> dict[str, Any]:
    return _executar_tool(
        "recomendar_cooperativas",
        {
            "tipo_material": tipo_material,
            "quantidade_minima_kg": quantidade_minima_kg,
            "raio_km": raio_km,
        },
        meta={"empresa_id": empresa_id} if empresa_id else None,
    )


# ==============================================================================
# Ferramentas: Base de Conhecimento e Normas PNRS
# ==============================================================================
@mcp.tool(
    name="faq_retriever",
    title="Consultar FAQ e base de conhecimento",
    description="Busca vetorial na base de conhecimento e FAQ de procedimentos operacionais e conformidade.",
    annotations=LEITURA,
)
def faq_retriever(
    pergunta: Annotated[str, Field(description="Pergunta ou termo operacional a ser pesquisado.")],
) -> dict[str, Any]:
    return _executar_tool("faq_retriever", {"pergunta": pergunta})


@mcp.tool(
    name="consultar_normas_pnrs",
    title="Consultar normas PNRS (Lei 12.305/2010)",
    description="Consulta as diretrizes legais, responsabilidade compartilhada e logística reversa na Política Nacional de Resíduos Sólidos.",
    annotations=LEITURA,
)
def consultar_normas_pnrs(
    tema: Annotated[str, Field(description="Tema ou artigo de interesse na PNRS.")],
) -> dict[str, Any]:
    container = _obter_container()
    res = container.mcp_server.read_resource("renovai://normas/pnrs")
    return {
        "status": "ok",
        "tema": tema,
        "conteudo": res.contents[0].text if res.contents else "Texto da PNRS não localizado.",
    }


# ==============================================================================
# Recursos MCP (@mcp.resource)
# ==============================================================================
@mcp.resource("renovai://normas/pnrs")
def resource_pnrs() -> str:
    """Texto normativo da Política Nacional de Resíduos Sólidos (Lei 12.305/2010)."""
    container = _obter_container()
    res = container.mcp_server.read_resource("renovai://normas/pnrs")
    return res.contents[0].text if res.contents else ""


@mcp.resource("renovai://telemetria/sre")
def resource_sre() -> str:
    """Painel de telemetria SRE, latência P95 e FinOps/ROI."""
    container = _obter_container()
    res = container.mcp_server.read_resource("renovai://telemetria/sre")
    return res.contents[0].text if res.contents else "{}"


@mcp.resource("renovai://telemetria/redis")
def resource_redis() -> str:
    """Estado em tempo real da fila de processamento e rankings Redis."""
    container = _obter_container()
    res = container.mcp_server.read_resource("renovai://telemetria/redis")
    return res.contents[0].text if res.contents else "{}"


# ==============================================================================
# Prompts Padronizados (@mcp.prompt)
# ==============================================================================
@mcp.prompt("diagnostico_estoque_cooperativa")
def prompt_diagnostico_estoque(cooperativa_id: str = "ativa", material_foco: str = "todos os recicláveis") -> str:
    """Prompt estruturado para diagnóstico crítico de giro de estoque e escoamento."""
    container = _obter_container()
    res = container.mcp_server.get_prompt("diagnostico_estoque_cooperativa", {"cooperativa_id": cooperativa_id, "material_foco": material_foco})
    return res.messages[0].content.text if res.messages else ""


@mcp.prompt("auditoria_rateio_financeiro")
def prompt_auditoria_rateio(cooperativa_id: str = "ativa", periodo: str = "mês recente") -> str:
    """Prompt estruturado para conciliação de receitas, despesas e rateio."""
    container = _obter_container()
    res = container.mcp_server.get_prompt("auditoria_rateio_financeiro", {"cooperativa_id": cooperativa_id, "periodo": periodo})
    return res.messages[0].content.text if res.messages else ""


@mcp.prompt("parecer_conformidade_pnrs")
def prompt_parecer_pnrs(tema: str = "responsabilidade compartilhada e logística reversa") -> str:
    """Prompt estruturado para enquadramento na PNRS (Lei 12.305/2010)."""
    container = _obter_container()
    res = container.mcp_server.get_prompt("parecer_conformidade_pnrs", {"tema": tema})
    return res.messages[0].content.text if res.messages else ""


# ==============================================================================
# Entrada Standalone CLI (transporte stdio)
# ==============================================================================
if __name__ == "__main__":
    from app.core.config import Settings

    settings = Settings()
    if not getattr(settings, "DATABASE_URL", None):
        print("[renovai-mcp] AVISO: DATABASE_URL ausente nas configurações", file=sys.stderr)

    mcp.run(transport="stdio")
