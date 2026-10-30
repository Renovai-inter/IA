"""Schemas Pydantic para o protocolo MCP (Model Context Protocol — JSON-RPC 2.0)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

MCP_PROTOCOL_VERSION = '2024-11-05'


class MCPJSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class MCPJSONRPCRequest(BaseModel):
    jsonrpc: Literal['2.0'] = '2.0'
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPJSONRPCResponse(BaseModel):
    jsonrpc: Literal['2.0'] = '2.0'
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[MCPJSONRPCError] = None


class MCPToolSpec(BaseModel):
    """Especificação de uma Tool exposta pelo servidor MCP (`tools/list`)."""

    name: str
    description: str
    inputSchema: Dict[str, Any] = Field(
        default_factory=lambda: {'type': 'object', 'properties': {}}
    )
    toolkit: Optional[str] = None
    domain: Optional[str] = None


class MCPContentBlock(BaseModel):
    type: Literal['text'] = 'text'
    text: str


class MCPCallToolResult(BaseModel):
    """Resultado padronizado de `tools/call` no protocolo MCP."""

    content: List[MCPContentBlock]
    isError: bool = False
    structuredContent: Optional[Any] = None


class MCPResourceSpec(BaseModel):
    """Especificação de um Resource contextual exposto pelo servidor MCP (`resources/list`)."""

    uri: str
    name: str
    description: str
    mimeType: str = 'application/json'


class MCPResourceContent(BaseModel):
    uri: str
    mimeType: str = 'application/json'
    text: str


class MCPReadResourceResult(BaseModel):
    contents: List[MCPResourceContent]


class MCPPromptArgument(BaseModel):
    name: str
    description: str
    required: bool = False


class MCPPromptSpec(BaseModel):
    """Especificação de um Prompt template exposto pelo servidor MCP (`prompts/list`)."""

    name: str
    description: str
    arguments: List[MCPPromptArgument] = Field(default_factory=list)


class MCPPromptMessage(BaseModel):
    role: Literal['user', 'assistant'] = 'user'
    content: MCPContentBlock


class MCPGetPromptResult(BaseModel):
    description: str
    messages: List[MCPPromptMessage]


class MCPServerInfo(BaseModel):
    name: str = 'renovai-mcp-server'
    version: str = '1.0.0'


class MCPCapabilities(BaseModel):
    tools: Dict[str, Any] = Field(default_factory=lambda: {'listChanged': False})
    resources: Dict[str, Any] = Field(
        default_factory=lambda: {'subscribe': False, 'listChanged': False}
    )
    prompts: Dict[str, Any] = Field(default_factory=lambda: {'listChanged': False})


class MCPInitializeResult(BaseModel):
    protocolVersion: str = MCP_PROTOCOL_VERSION
    capabilities: MCPCapabilities = Field(default_factory=MCPCapabilities)
    serverInfo: MCPServerInfo = Field(default_factory=MCPServerInfo)
