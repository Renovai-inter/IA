"""Implementação do protocolo Google Agent-to-Agent (A2A) com Grafo Principal Unificado
(Host Graph) acionando os Microsserviços A2A de Cooperativas (/a2a/cooperativa)
e de Empresas Compradoras (/a2a/empresa)."""

from app.a2a.schemas import (
    A2AMessage,
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    DataPart,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Task,
    TaskQueryParams,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)
from app.a2a.agent_card import (
    ROTAS_SERVICO_COOPERATIVA,
    ROTAS_SERVICO_EMPRESA,
    get_cooperativa_agent_card,
    get_empresa_agent_card,
)
from app.a2a.client import A2AClient
from app.a2a.server import (
    A2ACooperativaServer,
    A2AEmpresaServer,
    A2ASubgraphServer,
    router as a2a_router,
)

__all__ = [
    'A2AMessage',
    'AgentCapabilities',
    'AgentCard',
    'AgentProvider',
    'AgentSkill',
    'Artifact',
    'DataPart',
    'JSONRPCError',
    'JSONRPCRequest',
    'JSONRPCResponse',
    'Task',
    'TaskQueryParams',
    'TaskSendParams',
    'TaskState',
    'TaskStatus',
    'TextPart',
    'ROTAS_SERVICO_COOPERATIVA',
    'ROTAS_SERVICO_EMPRESA',
    'get_cooperativa_agent_card',
    'get_empresa_agent_card',
    'A2AClient',
    'A2ASubgraphServer',
    'A2ACooperativaServer',
    'A2AEmpresaServer',
    'a2a_router',
]
