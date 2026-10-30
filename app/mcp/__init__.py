"""Módulo MCP (Model Context Protocol — JSON-RPC 2.0) do Renovaí."""

from app.mcp.client import MCPClient
from app.mcp.schemas import (
    MCP_PROTOCOL_VERSION,
    MCPCallToolResult,
    MCPCapabilities,
    MCPContentBlock,
    MCPGetPromptResult,
    MCPInitializeResult,
    MCPJSONRPCError,
    MCPJSONRPCRequest,
    MCPJSONRPCResponse,
    MCPPromptArgument,
    MCPPromptMessage,
    MCPPromptSpec,
    MCPReadResourceResult,
    MCPResourceContent,
    MCPResourceSpec,
    MCPServerInfo,
    MCPToolSpec,
)
from app.mcp.server import MCPServer, router

__all__ = [
    'MCP_PROTOCOL_VERSION',
    'MCPCallToolResult',
    'MCPCapabilities',
    'MCPClient',
    'MCPContentBlock',
    'MCPGetPromptResult',
    'MCPInitializeResult',
    'MCPJSONRPCError',
    'MCPJSONRPCRequest',
    'MCPJSONRPCResponse',
    'MCPPromptArgument',
    'MCPPromptMessage',
    'MCPPromptSpec',
    'MCPReadResourceResult',
    'MCPResourceContent',
    'MCPResourceSpec',
    'MCPServer',
    'MCPServerInfo',
    'MCPToolSpec',
    'router',
]
