from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.a2a.agent_card import get_cooperativa_agent_card, get_empresa_agent_card
from app.a2a.schemas import (
    A2AMessage,
    Artifact,
    DataPart,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Task,
    TaskQueryParams,
    TaskSendParams,
    TaskStatus,
    TextPart,
)
from app.graph.nodes import make_repo_backed_runners
from app.graph.registro import RegistroEspecialista
from app.repository.base import Repository
from app.repository.postgresql.perfil_repository import PerfilContext, _to_uuid
from app.core.observabilidade import medir_span

log = logging.getLogger('renovai.a2a')

router = APIRouter(tags=['a2a'])


def _reconstruir_mensagens_langchain(
    mensagens_serializadas: List[Dict[str, Any]],
    texto_fallback: str,
) -> List[BaseMessage]:
    if not mensagens_serializadas:
        return [HumanMessage(content=texto_fallback)] if texto_fallback else []

    reconstruidas: List[BaseMessage] = []
    for item in mensagens_serializadas:
        role = item.get('role') or item.get('type') or 'human'
        content = str(item.get('content') or '')
        if role in ('human', 'user', 'usuario'):
            reconstruidas.append(HumanMessage(content=content))
        else:
            reconstruidas.append(AIMessage(content=content))
    return reconstruidas


class A2ASubgraphServer:
    """Servidor Google A2A (JSON-RPC 2.0) que hospeda o subgrafo de especialistas de um
    domínio específico (`COOPERATIVA` ou `EMPRESA`)."""

    dominio: Literal['COOPERATIVA', 'EMPRESA'] = 'COOPERATIVA'
    _MAX_TASKS_EM_MEMORIA: int = 100

    def __init__(
        self,
        specialists: Optional[List[RegistroEspecialista]] = None,
        repositories: Optional[Dict[str, Repository]] = None,
    ):
        self.repositories = repositories or {}
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._runners_por_skill: Dict[str, tuple[RegistroEspecialista, Any, Any]] = {}
        for registro in (specialists or []):
            self.registrar_especialista(registro)

    def _registrar_task(self, task: Task) -> None:
        with self._lock:
            self._tasks[task.id] = task
            while len(self._tasks) > self._MAX_TASKS_EM_MEMORIA:
                oldest_key = next(iter(self._tasks))
                self._tasks.pop(oldest_key, None)

    def registrar_especialista(self, registro: RegistroEspecialista) -> None:
        agent_repos = {
            nome: self.repositories[nome]
            for nome in registro.repositories
            if nome in self.repositories
        }
        sync_fn, async_fn = make_repo_backed_runners(
            registro.agente,
            agent_repos,
            exigir_empresa=(self.dominio == 'EMPRESA'),
        )
        self._runners_por_skill[registro.rota] = (registro, sync_fn, async_fn)

    @property
    def skills_registradas(self) -> List[str]:
        return list(self._runners_por_skill.keys())

    def _extrair_contexto_a2a(
        self,
        params: TaskSendParams,
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        dados: Dict[str, Any] = dict(params.metadata or {})
        if params.message.metadata:
            dados.update(params.message.metadata)

        textos: List[str] = []
        for part in params.message.parts:
            if isinstance(part, DataPart):
                dados.update(part.data or {})
            elif isinstance(part, TextPart):
                textos.append(part.text)

        pergunta_fallback = '\n'.join(textos)
        skill = str(dados.get('skill') or dados.get('rota') or '').strip()

        coop_uuid = _to_uuid(dados.get('cooperativa_id'))
        emp_uuid = _to_uuid(dados.get('empresa_id'))

        perfil_ctx: Optional[PerfilContext] = None
        if coop_uuid is not None or emp_uuid is not None:
            perfil_ctx = PerfilContext.from_request(
                cooperativa_id=coop_uuid,
                empresa_id=emp_uuid,
            )

        mensagens_lc = _reconstruir_mensagens_langchain(
            dados.get('messages') or [],
            pergunta_fallback,
        )

        entidade_id = str(coop_uuid or emp_uuid or dados.get('perfil_id') or '')
        sub_state: Dict[str, Any] = {
            'messages': mensagens_lc,
            'cooperativa_id': coop_uuid,
            'empresa_id': emp_uuid,
            'perfil_ctx': perfil_ctx,
            'proximos_agentes': [skill] if skill else [],
        }
        sub_config: Dict[str, Any] = {
            'configurable': {
                'thread_id': params.sessionId or dados.get('thread_id') or params.id,
                'user_id': dados.get('user_id'),
                'cooperativa_id': str(coop_uuid) if coop_uuid else None,
                'empresa_id': str(emp_uuid) if emp_uuid else None,
                'perfil_id': entidade_id or None,
            }
        }
        return skill, sub_state, sub_config

    def _montar_task_concluida(
        self,
        params: TaskSendParams,
        skill: str,
        registro: RegistroEspecialista,
        resultado_no: Dict[str, Any],
    ) -> Task:
        nome_agente = getattr(registro.agente, 'nome_agente', registro.node_name)
        agentes_chamados = list(resultado_no.get('agentes_chamados') or [nome_agente])
        outputs_brutos = resultado_no.get('especialista_outputs') or {}

        outputs_serializados: Dict[str, Dict[str, Any]] = {}
        texto_principal = ''
        for chave, out_obj in outputs_brutos.items():
            conteudo = getattr(out_obj, 'conteudo', None)
            if conteudo is None and isinstance(out_obj, dict):
                conteudo = out_obj.get('conteudo', '')
            fonte = getattr(out_obj, 'fonte', None)
            if fonte is None and isinstance(out_obj, dict):
                fonte = out_obj.get('fonte', chave)
            outputs_serializados[chave] = {
                'conteudo': str(conteudo or ''),
                'fonte': str(fonte or chave),
            }
            if not texto_principal and conteudo:
                texto_principal = str(conteudo)

        if not texto_principal:
            msgs_saida = resultado_no.get('messages') or []
            if msgs_saida:
                ult = msgs_saida[-1]
                texto_principal = (
                    ult.get('content', '')
                    if isinstance(ult, dict)
                    else str(getattr(ult, 'content', ''))
                )

        payload_artifact = {
            'dominio_a2a': self.dominio,
            'skill': skill,
            'node_name': registro.node_name,
            'nome_agente': nome_agente,
            'conteudo': texto_principal,
            'agentes_chamados': agentes_chamados,
            'especialista_outputs': outputs_serializados,
        }

        resposta_msg = A2AMessage(
            role='agent',
            parts=[TextPart(text=texto_principal)],
        )
        artifact = Artifact(
            name=f'especialista_{skill}',
            description=f"Saída do especialista '{nome_agente}' via protocolo Google A2A ({self.dominio})",
            parts=[
                DataPart(data=payload_artifact),
                TextPart(text=texto_principal),
            ],
        )
        task = Task(
            id=params.id,
            sessionId=params.sessionId,
            status=TaskStatus(state='completed', message=resposta_msg),
            artifacts=[artifact],
            history=[params.message, resposta_msg],
            metadata={'dominio': self.dominio, 'skill': skill, 'nome_agente': nome_agente},
        )
        self._registrar_task(task)
        return task

    def _montar_task_falha(self, params: TaskSendParams, mensagem_erro: str) -> Task:
        msg_erro = A2AMessage(
            role='agent',
            parts=[TextPart(text=mensagem_erro)],
        )
        task = Task(
            id=params.id,
            sessionId=params.sessionId,
            status=TaskStatus(state='failed', message=msg_erro),
            history=[params.message, msg_erro],
        )
        self._registrar_task(task)
        return task

    def handle_task_send(self, params: TaskSendParams) -> Task:
        skill, sub_state, sub_config = self._extrair_contexto_a2a(params)
        alvo = self._runners_por_skill.get(skill)
        if alvo is None:
            return self._montar_task_falha(
                params,
                f"Skill '{skill}' não registrada no serviço A2A {self.dominio}. "
                f'Skills disponíveis: {self.skills_registradas}',
            )
        registro, sync_fn, _ = alvo
        with medir_span(f'a2a_server_{self.dominio.lower()}_{skill}', categoria='a2a_server'):
            resultado_no = sync_fn(sub_state, sub_config)
        return self._montar_task_concluida(params, skill, registro, resultado_no)

    async def ahandle_task_send(self, params: TaskSendParams) -> Task:
        skill, sub_state, sub_config = self._extrair_contexto_a2a(params)
        alvo = self._runners_por_skill.get(skill)
        if alvo is None:
            return self._montar_task_falha(
                params,
                f"Skill '{skill}' não registrada no serviço A2A {self.dominio}."
                f'Skills disponíveis: {self.skills_registradas}',
            )
        registro, _, async_fn = alvo
        with medir_span(f'a2a_server_{self.dominio.lower()}_{skill}', categoria='a2a_server'):
            resultado_no = await async_fn(sub_state, sub_config)
        return self._montar_task_concluida(params, skill, registro, resultado_no)

    def _processar_get_ou_cancel(self, rpc_req: JSONRPCRequest) -> JSONRPCResponse:
        if rpc_req.method == 'tasks/get':
            query = TaskQueryParams.model_validate(rpc_req.params)
            with self._lock:
                task = self._tasks.get(query.id)
            if task is None:
                return JSONRPCResponse(
                    id=rpc_req.id,
                    error=JSONRPCError(code=-32001, message=f'Task {query.id} não encontrada.'),
                )
            return JSONRPCResponse(id=rpc_req.id, result=task)

        if rpc_req.method == 'tasks/cancel':
            query = TaskQueryParams.model_validate(rpc_req.params)
            with self._lock:
                task = self._tasks.get(query.id)
            if task is None:
                return JSONRPCResponse(
                    id=rpc_req.id,
                    error=JSONRPCError(code=-32001, message=f'Task {query.id} não encontrada.'),
                )
            task.status = TaskStatus(state='canceled')
            return JSONRPCResponse(id=rpc_req.id, result=task)

        return JSONRPCResponse(
            id=rpc_req.id,
            error=JSONRPCError(code=-32601, message=f'Método A2A não suportado: {rpc_req.method}'),
        )

    def handle_jsonrpc(self, rpc_req: JSONRPCRequest) -> JSONRPCResponse:
        try:
            if rpc_req.method == 'tasks/send':
                send_params = TaskSendParams.model_validate(rpc_req.params)
                task = self.handle_task_send(send_params)
                return JSONRPCResponse(id=rpc_req.id, result=task)
            return self._processar_get_ou_cancel(rpc_req)
        except Exception as exc:
            log.exception('Erro ao processar requisição JSON-RPC A2A (%s)', self.dominio)
            return JSONRPCResponse(
                id=rpc_req.id,
                error=JSONRPCError(code=-32602, message=str(exc)),
            )

    async def ahandle_jsonrpc(self, rpc_req: JSONRPCRequest) -> JSONRPCResponse:
        try:
            if rpc_req.method == 'tasks/send':
                send_params = TaskSendParams.model_validate(rpc_req.params)
                task = await self.ahandle_task_send(send_params)
                return JSONRPCResponse(id=rpc_req.id, result=task)
            return self._processar_get_ou_cancel(rpc_req)
        except Exception as exc:
            log.exception('Erro assíncrono ao processar requisição JSON-RPC A2A (%s)', self.dominio)
            return JSONRPCResponse(
                id=rpc_req.id,
                error=JSONRPCError(code=-32602, message=str(exc)),
            )


class A2ACooperativaServer(A2ASubgraphServer):
    """Microsserviço A2A de Cooperativas (/a2a/cooperativa) — hospeda os especialistas
    de `estoque`, `coleta`, `financeiro` e `logistica` operando por `cooperativa_id`."""
    dominio: Literal['COOPERATIVA', 'EMPRESA'] = 'COOPERATIVA'


class A2AEmpresaServer(A2ASubgraphServer):
    """Microsserviço A2A de Empresas Compradoras (/a2a/empresa) — hospeda o especialista
    de `pedidos` (`recomendacao`) operando por `empresa_id`."""
    dominio: Literal['COOPERATIVA', 'EMPRESA'] = 'EMPRESA'


def _obter_cooperativa_server(request: Request) -> A2ACooperativaServer:
    container = getattr(request.app.state, 'container', None)
    if container is not None and getattr(container, 'a2a_cooperativa_server', None) is not None:
        return container.a2a_cooperativa_server
    if container is not None and getattr(container, 'a2a_server', None) is not None:
        return container.a2a_server
    return A2ACooperativaServer()


def _obter_empresa_server(request: Request) -> A2AEmpresaServer:
    container = getattr(request.app.state, 'container', None)
    if container is not None and getattr(container, 'a2a_empresa_server', None) is not None:
        return container.a2a_empresa_server
    return A2AEmpresaServer()


@router.get('/.well-known/agent.json', response_model=None)
@router.get('/a2a/cooperativa/.well-known/agent.json', response_model=None)
async def cooperativa_agent_card(request: Request):
    base_url = str(request.base_url).rstrip('/') + '/a2a/cooperativa'
    return get_cooperativa_agent_card(base_url=base_url)


@router.get('/a2a/empresa/.well-known/agent.json', response_model=None)
async def empresa_agent_card(request: Request):
    base_url = str(request.base_url).rstrip('/') + '/a2a/empresa'
    return get_empresa_agent_card(base_url=base_url)


@router.post('/a2a/cooperativa', response_model=JSONRPCResponse)
async def cooperativa_a2a_endpoint(request: Request, rpc_req: JSONRPCRequest) -> JSONRPCResponse:
    server = _obter_cooperativa_server(request)
    return await server.ahandle_jsonrpc(rpc_req)


@router.post('/a2a/empresa', response_model=JSONRPCResponse)
async def empresa_a2a_endpoint(request: Request, rpc_req: JSONRPCRequest) -> JSONRPCResponse:
    server = _obter_empresa_server(request)
    return await server.ahandle_jsonrpc(rpc_req)
