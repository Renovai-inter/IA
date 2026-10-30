"""Testes unitários para o Servidor e Cliente MCP (Model Context Protocol — JSON-RPC 2.0).

Cobre:
1. Contratos e Schemas MCP (RFC 2024-11-05);
2. Descoberta de Tools (`tools/list`), Resources (`resources/list`) e Prompts (`prompts/list`);
3. Invocação JSON-RPC 2.0 (`initialize`, `tools/call`, `resources/read`, `prompts/get`);
4. Tratamento de Erros e Validação JSON-RPC 2.0 (-32601 Method Not Found, -32602 Invalid Params);
5. Instrumentação de StructuredTools via `wrap_toolkit_tools` com spans de telemetria;
6. Cliente MCP (`MCPClient`) em modo in-process;
7. Endpoints FastAPI (`/.well-known/mcp.json`, `/mcp/manifest`, `/mcp/tools`, `/mcp/resources`, `/mcp/prompts`, `POST /mcp`, `/monitor/mcp`).
"""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.observabilidade import Observabilidade
from app.main import app
from app.mcp.client import MCPClient
from app.mcp.schemas import (
    MCP_PROTOCOL_VERSION,
    MCPCallToolResult,
    MCPGetPromptResult,
    MCPInitializeResult,
    MCPJSONRPCRequest,
    MCPJSONRPCResponse,
    MCPPromptSpec,
    MCPReadResourceResult,
    MCPResourceSpec,
    MCPToolSpec,
)
from app.mcp.server import MCPServer
from tests.support import StubRepository


# ==============================================================================
# Suporte / Fixtures de Teste
# ==============================================================================
class _EntradaCalculo(BaseModel):
    peso_kg: float = Field(..., description="Peso em quilogramas")
    tipo_material: str = Field(default="PET", description="Tipo do resíduo")


def _fake_calcular_valor(peso_kg: float, tipo_material: str = "PET") -> Dict[str, Any]:
    fator = 2.5 if tipo_material.upper() == "PET" else 1.0
    return {"status": "ok", "peso_kg": peso_kg, "tipo": tipo_material, "total_reais": peso_kg * fator}


@pytest.fixture
def fake_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_fake_calcular_valor,
        name="calcular_valor_residuo",
        description="Calcula o valor estimado de um lote de recicláveis.",
        args_schema=_EntradaCalculo,
    )


@pytest.fixture
def mcp_server(fake_tool: StructuredTool) -> MCPServer:
    obs = Observabilidade()
    srv = MCPServer(
        toolkits={},
        repositories={
            "material_repository": StubRepository(snapshot=[]),
            "estoque_repository": StubRepository(snapshot=[]),
        },
        observabilidade=obs,
        pnrs_path=None,
    )
    srv.registrar_toolkit("material_estoque_toolkit", [fake_tool])
    return srv


@pytest.fixture
def mcp_client(mcp_server: MCPServer) -> MCPClient:
    return MCPClient(server=mcp_server)


# ==============================================================================
# 1. Descoberta de Tools, Resources e Prompts
# ==============================================================================
def test_mcp_server_descoberta_tools(mcp_server: MCPServer, fake_tool: StructuredTool):
    tools = mcp_server.listar_tools()
    assert len(tools) >= 1
    nomes = [t.name for t in tools]
    assert "calcular_valor_residuo" in nomes

    spec = next(t for t in tools if t.name == "calcular_valor_residuo")
    assert spec.domain == "COOPERATIVA"
    assert spec.toolkit == "material_estoque_toolkit"
    assert "properties" in spec.inputSchema
    assert "peso_kg" in spec.inputSchema["properties"]

    # Filtro por domínio
    tools_coop = mcp_server.listar_tools(domain="COOPERATIVA")
    assert any(t.name == "calcular_valor_residuo" for t in tools_coop)
    tools_emp = mcp_server.listar_tools(domain="EMPRESA")
    assert not any(t.name == "calcular_valor_residuo" for t in tools_emp)


def test_mcp_server_descoberta_resources(mcp_server: MCPServer):
    resources = mcp_server.listar_resources()
    uris = [r.uri for r in resources]
    assert "renovai://normas/pnrs" in uris
    assert "renovai://telemetria/sre" in uris
    assert "renovai://telemetria/redis" in uris
    assert "renovai://cooperativa/{cooperativa_id}/estoque" in uris


def test_mcp_server_descoberta_prompts(mcp_server: MCPServer):
    prompts = mcp_server.listar_prompts()
    nomes = [p.name for p in prompts]
    assert "parecer_conformidade_pnrs" in nomes
    assert "diagnostico_estoque_cooperativa" in nomes
    assert "auditoria_rateio_financeiro" in nomes


# ==============================================================================
# 2. Invocação JSON-RPC 2.0 (initialize, tools/call, resources/read, prompts/get)
# ==============================================================================
def test_mcp_server_rpc_initialize(mcp_server: MCPServer):
    req = MCPJSONRPCRequest(id="req-1", method="initialize", params={})
    res = mcp_server.handle_rpc(req)

    assert res.error is None
    assert res.id == "req-1"
    assert res.result is not None
    init_res = MCPInitializeResult.model_validate(res.result)
    assert init_res.protocolVersion == MCP_PROTOCOL_VERSION
    assert init_res.serverInfo.name == "renovai-mcp-server"
    assert init_res.capabilities.tools is not None
    assert init_res.capabilities.resources is not None
    assert init_res.capabilities.prompts is not None


def test_mcp_server_rpc_tools_call(mcp_server: MCPServer):
    req = MCPJSONRPCRequest(
        id="req-2",
        method="tools/call",
        params={
            "name": "calcular_valor_residuo",
            "arguments": {"peso_kg": 100.0, "tipo_material": "PET"},
        },
    )
    res = mcp_server.handle_rpc(req)

    assert res.error is None
    assert res.result is not None
    call_res = MCPCallToolResult.model_validate(res.result)
    assert not call_res.isError
    assert len(call_res.content) >= 1
    assert "total_reais" in call_res.content[0].text
    assert "250.0" in call_res.content[0].text


def test_mcp_server_rpc_resources_read(mcp_server: MCPServer):
    req = MCPJSONRPCRequest(
        id="req-3",
        method="resources/read",
        params={"uri": "renovai://telemetria/sre"},
    )
    res = mcp_server.handle_rpc(req)

    assert res.error is None
    assert res.result is not None
    read_res = MCPReadResourceResult.model_validate(res.result)
    assert len(read_res.contents) >= 1
    assert read_res.contents[0].uri == "renovai://telemetria/sre"
    assert read_res.contents[0].mimeType == "application/json"


def test_mcp_server_rpc_prompts_get(mcp_server: MCPServer):
    req = MCPJSONRPCRequest(
        id="req-4",
        method="prompts/get",
        params={
            "name": "parecer_conformidade_pnrs",
            "arguments": {
                "tema": "Logística reversa de embalagens plásticas",
            },
        },
    )
    res = mcp_server.handle_rpc(req)

    assert res.error is None
    assert res.result is not None
    prompt_res = MCPGetPromptResult.model_validate(res.result)
    assert len(prompt_res.messages) >= 1
    assert "Logística reversa de embalagens plásticas" in prompt_res.messages[0].content.text
    assert "Lei nº 12.305/2010" in prompt_res.messages[0].content.text


# ==============================================================================
# 3. Tratamento de Erros JSON-RPC 2.0 (-32601, -32602)
# ==============================================================================
def test_mcp_server_rpc_metodo_inexistente(mcp_server: MCPServer):
    req = MCPJSONRPCRequest(id="err-1", method="metodo_desconhecido", params={})
    res = mcp_server.handle_rpc(req)

    assert res.result is None
    assert res.error is not None
    assert res.error.code == -32601
    assert "desconhecido" in res.error.message


def test_mcp_server_rpc_parametros_invalidos(mcp_server: MCPServer):
    req = MCPJSONRPCRequest(id="err-2", method="tools/call", params={})
    res = mcp_server.handle_rpc(req)

    assert res.result is None
    assert res.error is not None
    assert res.error.code == -32602


# ==============================================================================
# 4. Instrumentação de StructuredTools via wrap_toolkit_tools
# ==============================================================================
def test_mcp_server_wrap_toolkit_tools(fake_tool: StructuredTool):
    obs = Observabilidade()
    srv = MCPServer(observabilidade=obs)
    wrapped_tools = srv.wrap_toolkit_tools("material_estoque_toolkit", [fake_tool])

    assert len(wrapped_tools) == 1
    w_tool = wrapped_tools[0]
    assert w_tool.name == "calcular_valor_residuo"

    # Invocação direta da tool embrulhada
    resultado = w_tool.invoke({"peso_kg": 200.0, "tipo_material": "PET"})
    assert resultado["status"] == "ok"
    assert resultado["total_reais"] == 500.0

    # Verifica telemetria coletada no servidor MCP
    painel = srv.obter_painel_mcp()
    assert painel["resumo"]["total_chamadas_tools"] == 1
    assert painel["resumo"]["total_erros_tools"] == 0
    stats = next(s for s in painel["por_tool"] if s["tool"] == "calcular_valor_residuo")
    assert stats["chamadas"] == 1
    assert stats["toolkit"] == "material_estoque_toolkit"


# ==============================================================================
# 5. Cliente MCP (MCPClient) in-process
# ==============================================================================
def test_mcp_client_in_process(mcp_client: MCPClient):
    init = mcp_client.initialize()
    assert init.protocolVersion == MCP_PROTOCOL_VERSION

    tools = mcp_client.list_tools()
    assert any(t.name == "calcular_valor_residuo" for t in tools)

    call_res = mcp_client.call_tool(
        name="calcular_valor_residuo",
        arguments={"peso_kg": 50.0, "tipo_material": "PET"},
    )
    assert not call_res.isError
    assert "125.0" in call_res.content[0].text

    resources = mcp_client.list_resources()
    assert len(resources) >= 3

    read_res = mcp_client.read_resource("renovai://telemetria/sre")
    assert len(read_res.contents) >= 1

    prompts = mcp_client.list_prompts()
    assert len(prompts) >= 3

    prompt_res = mcp_client.get_prompt(
        "diagnostico_estoque_cooperativa",
        arguments={"cooperativa_id": "coop-sp-1", "material_foco": "Papelão"},
    )
    assert "coop-sp-1" in prompt_res.messages[0].content.text
    assert "Papelão" in prompt_res.messages[0].content.text


# ==============================================================================
# 6. Endpoints HTTP FastAPI (TestClient)
# ==============================================================================
def test_mcp_endpoints_http():
    with TestClient(app) as client:
        # 1. Manifest / Discovery
        r_manifest = client.get("/.well-known/mcp.json")
        assert r_manifest.status_code == 200
        data_manifest = r_manifest.json()
        assert data_manifest["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert "tools" in data_manifest
        assert "resources" in data_manifest
        assert "prompts" in data_manifest

        # 2. GET /mcp/tools
        r_tools = client.get("/mcp/tools")
        assert r_tools.status_code == 200
        data_tools = r_tools.json()
        assert data_tools["count"] >= 1
        assert "stats" in data_tools

        # 3. GET /mcp/resources
        r_resources = client.get("/mcp/resources")
        assert r_resources.status_code == 200
        assert r_resources.json()["count"] >= 3

        # 4. GET /mcp/prompts
        r_prompts = client.get("/mcp/prompts")
        assert r_prompts.status_code == 200
        assert r_prompts.json()["count"] >= 3

        # 5. POST /mcp (JSON-RPC 2.0 initialize)
        rpc_payload = {
            "jsonrpc": "2.0",
            "id": "http-test-1",
            "method": "initialize",
            "params": {},
        }
        r_rpc = client.post("/mcp", json=rpc_payload)
        assert r_rpc.status_code == 200
        data_rpc = r_rpc.json()
        assert data_rpc["id"] == "http-test-1"
        assert data_rpc["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION

        # 6. GET /monitor/mcp
        r_monitor_mcp = client.get("/monitor/mcp")
        assert r_monitor_mcp.status_code == 200
        data_mon = r_monitor_mcp.json()
        assert "protocolo" in data_mon
        assert "resumo" in data_mon
        assert "por_tool" in data_mon
