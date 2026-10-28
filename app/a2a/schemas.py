from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentProvider(BaseModel):
    organization: str = 'Renovaí'
    url: str = 'https://renovai.org.br'


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = True


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    inputModes: List[str] = Field(default_factory=lambda: ['application/json', 'text/plain'])
    outputModes: List[str] = Field(default_factory=lambda: ['application/json', 'text/plain'])


class AgentCard(BaseModel):
    """Agent Card oficial segundo a especificação Google A2A."""
    name: str
    description: str
    url: str
    provider: AgentProvider = Field(default_factory=AgentProvider)
    version: str = '1.0.0'
    protocolVersion: str = '0.2.1'
    defaultInputModes: List[str] = Field(default_factory=lambda: ['application/json', 'text/plain'])
    defaultOutputModes: List[str] = Field(default_factory=lambda: ['application/json', 'text/plain'])
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = Field(default_factory=list)


class TextPart(BaseModel):
    type: Literal['text'] = 'text'
    text: str


class DataPart(BaseModel):
    type: Literal['data'] = 'data'
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


Part = Annotated[Union[TextPart, DataPart], Field(discriminator='type')]


class A2AMessage(BaseModel):
    role: Literal['user', 'agent']
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None


TaskState = Literal[
    'submitted',
    'working',
    'input-required',
    'completed',
    'canceled',
    'failed',
    'unknown',
]


class TaskStatus(BaseModel):
    state: TaskState
    message: Optional[A2AMessage] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Artifact(BaseModel):
    name: str
    description: Optional[str] = None
    parts: List[Part]
    index: int = 0


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    sessionId: Optional[str] = None
    status: TaskStatus
    artifacts: List[Artifact] = Field(default_factory=list)
    history: List[A2AMessage] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class TaskSendParams(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    sessionId: Optional[str] = None
    message: A2AMessage
    metadata: Optional[Dict[str, Any]] = None


class TaskQueryParams(BaseModel):
    id: str


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal['2.0'] = '2.0'
    id: Union[str, int]
    method: Literal['tasks/send', 'tasks/get', 'tasks/cancel']
    params: Dict[str, Any] = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal['2.0'] = '2.0'
    id: Optional[Union[str, int]] = None
    result: Optional[Task] = None
    error: Optional[JSONRPCError] = None
