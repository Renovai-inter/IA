"""Cliente MCP (Model Context Protocol — JSON-RPC 2.0) do ecossistema Renovaí.

Permite que agentes e serviços descubram e invoquem Tools (`tools/list`, `tools/call`),
Resources (`resources/list`, `resources/read`) e Prompts (`prompts/list`, `prompts/get`)
tanto em modo local (in-process) quanto via transporte HTTP JSON-RPC (`POST /mcp`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from app.mcp.schemas import (
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

log = logging.getLogger('renovai.mcp.client')


class MCPClient:
    """Cliente do protocolo MCP para descoberta e execução de Tools, Resources e Prompts."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        server: Optional[MCPServer] = None,
        timeout: float = 30.0,
    ):
        self.server_url = (server_url or '').rstrip('/') or None
        self.server = server or MCPServer()
        self.timeout = timeout
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _obter_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    def _obter_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def close(self) -> None:
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()

    async def aclose(self) -> None:
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()

    def _send_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> MCPJSONRPCResponse:
        rpc_req = MCPJSONRPCRequest(
            id=str(uuid4()),
            method=method,
            params=params or {},
        )
        if self.server_url:
            client = self._obter_sync_client()
            resp = client.post(f'{self.server_url}/mcp', json=rpc_req.model_dump())
            resp.raise_for_status()
            return MCPJSONRPCResponse.model_validate(resp.json())

        return self.server.handle_rpc(rpc_req)

    async def _asend_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> MCPJSONRPCResponse:
        rpc_req = MCPJSONRPCRequest(
            id=str(uuid4()),
            method=method,
            params=params or {},
        )
        if self.server_url:
            client = self._obter_async_client()
            resp = await client.post(f'{self.server_url}/mcp', json=rpc_req.model_dump())
            resp.raise_for_status()
            return MCPJSONRPCResponse.model_validate(resp.json())

        return await self.server.ahandle_rpc(rpc_req)

    # --------------------------------------------------------------------------
    # Operações MCP Padronizadas
    # --------------------------------------------------------------------------
    def initialize(self) -> MCPInitializeResult:
        rpc_resp = self._send_rpc('initialize')
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP initialize: {rpc_resp.error.message}')
        return MCPInitializeResult.model_validate(rpc_resp.result)

    def list_tools(self, domain: Optional[str] = None) -> List[MCPToolSpec]:
        params = {'domain': domain} if domain else {}
        rpc_resp = self._send_rpc('tools/list', params=params)
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP tools/list: {rpc_resp.error.message}')
        tools_raw = (rpc_resp.result or {}).get('tools') or []
        return [MCPToolSpec.model_validate(t) for t in tools_raw]

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> MCPCallToolResult:
        if not self.server_url and config is not None:
            return self.server.call_tool(name=name, arguments=arguments, meta=meta, config=config)

        rpc_resp = self._send_rpc(
            'tools/call',
            params={'name': name, 'arguments': arguments or {}, '_meta': meta or {}},
        )
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP tools/call ({name}): {rpc_resp.error.message}')
        return MCPCallToolResult.model_validate(rpc_resp.result)

    async def acall_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> MCPCallToolResult:
        if not self.server_url and config is not None:
            return await self.server.acall_tool(name=name, arguments=arguments, meta=meta, config=config)

        rpc_resp = await self._asend_rpc(
            'tools/call',
            params={'name': name, 'arguments': arguments or {}, '_meta': meta or {}},
        )
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP tools/call ({name}): {rpc_resp.error.message}')
        return MCPCallToolResult.model_validate(rpc_resp.result)

    def list_resources(self) -> List[MCPResourceSpec]:
        rpc_resp = self._send_rpc('resources/list')
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP resources/list: {rpc_resp.error.message}')
        res_raw = (rpc_resp.result or {}).get('resources') or []
        return [MCPResourceSpec.model_validate(r) for r in res_raw]

    def read_resource(self, uri: str) -> MCPReadResourceResult:
        rpc_resp = self._send_rpc('resources/read', params={'uri': uri})
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP resources/read ({uri}): {rpc_resp.error.message}')
        return MCPReadResourceResult.model_validate(rpc_resp.result)

    def list_prompts(self) -> List[MCPPromptSpec]:
        rpc_resp = self._send_rpc('prompts/list')
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP prompts/list: {rpc_resp.error.message}')
        prompts_raw = (rpc_resp.result or {}).get('prompts') or []
        return [MCPPromptSpec.model_validate(p) for p in prompts_raw]

    def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPGetPromptResult:
        rpc_resp = self._send_rpc('prompts/get', params={'name': name, 'arguments': arguments or {}})
        if rpc_resp.error:
            raise RuntimeError(f'Erro MCP prompts/get ({name}): {rpc_resp.error.message}')
        return MCPGetPromptResult.model_validate(rpc_resp.result)

    def get_toolkit_tools(
        self,
        toolkit_name: str,
        tools: List[StructuredTool],
    ) -> List[StructuredTool]:
        """Conecta e instrumenta as StructuredTools de um toolkit através do servidor MCP."""
        return self.server.wrap_toolkit_tools(toolkit_name, tools)
