from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import httpx

from app.a2a.agent_card import (
    ROTAS_SERVICO_EMPRESA,
    get_cooperativa_agent_card,
    get_empresa_agent_card,
)
from app.a2a.schemas import (
    A2AMessage,
    AgentCard,
    DataPart,
    JSONRPCRequest,
    JSONRPCResponse,
    Task,
    TextPart,
)
from app.a2a.server import A2ACooperativaServer, A2AEmpresaServer
from app.graph.state import EspecialistaOutput
from app.core.observabilidade import medir_span

log = logging.getLogger('renovai.a2a.client')

ServicoA2A = Literal['cooperativa', 'empresa']


def _serializar_mensagens(messages: List[Any]) -> List[Dict[str, str]]:
    serializadas: List[Dict[str, str]] = []
    for m in (messages or []):
        if isinstance(m, dict):
            role = str(m.get('role') or m.get('type') or 'human')
            content = str(m.get('content') or '')
        else:
            role = str(getattr(m, 'type', 'human'))
            raw_content = getattr(m, 'content', '')
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                content = ''.join(
                    b.get('text', '') for b in raw_content if isinstance(b, dict) and b.get('type') == 'text'
                )
            else:
                content = str(raw_content)
        serializadas.append({'role': role, 'content': content})
    return serializadas


class A2AClient:
    """Cliente do protocolo Google A2A utilizado pelo Grafo Principal (Host Graph / Roteador)
    para acionar os endpoints dos Microsserviços A2A (`/a2a/cooperativa` e `/a2a/empresa`)
    de acordo com a rota classificada e o `cooperativa_id` / `empresa_id` da requisição."""

    def __init__(
        self,
        cooperativa_url: Optional[str] = None,
        empresa_url: Optional[str] = None,
        cooperativa_server: Optional[A2ACooperativaServer] = None,
        empresa_server: Optional[A2AEmpresaServer] = None,
        timeout: float = 30.0,
    ):
        self.cooperativa_url = (cooperativa_url or '').rstrip('/') or None
        self.empresa_url = (empresa_url or '').rstrip('/') or None
        self.cooperativa_server = cooperativa_server
        self.empresa_server = empresa_server
        self.timeout = timeout
        self._cards_cache: Dict[str, AgentCard] = {}

    def discover_agent_card(self, servico: ServicoA2A = 'cooperativa') -> AgentCard:
        """Descobre o Agent Card do microsserviço (`cooperativa` ou `empresa`) via `/.well-known/agent.json`."""
        if servico in self._cards_cache:
            return self._cards_cache[servico]

        url_base = self.cooperativa_url if servico == 'cooperativa' else self.empresa_url
        if url_base:
            url = f'{url_base}/.well-known/agent.json'
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                card = AgentCard.model_validate(resp.json())
                self._cards_cache[servico] = card
                return card

        card = (
            get_cooperativa_agent_card()
            if servico == 'cooperativa'
            else get_empresa_agent_card()
        )
        self._cards_cache[servico] = card
        return card

    def resolver_servico_por_skill(self, skill: str) -> ServicoA2A:
        """Determina qual microsserviço A2A (`cooperativa` ou `empresa`) hospeda a skill/rota."""
        card_emp = self.discover_agent_card('empresa')
        if any(s.id == skill for s in card_emp.skills) or skill in ROTAS_SERVICO_EMPRESA:
            return 'empresa'
        if self.empresa_server and skill in self.empresa_server.skills_registradas:
            return 'empresa'
        return 'cooperativa'

    def _montar_rpc_request(
        self,
        skill: str,
        servico: ServicoA2A,
        cooperativa_id: Optional[str] = None,
        empresa_id: Optional[str] = None,
        messages: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        callbacks: Optional[Any] = None,
    ) -> JSONRPCRequest:
        msgs_serializadas = _serializar_mensagens(messages or [])
        ultima_pergunta = ''
        for item in reversed(msgs_serializadas):
            if item['role'] in ('human', 'user', 'usuario'):
                ultima_pergunta = item['content']
                break
        if not ultima_pergunta and msgs_serializadas:
            ultima_pergunta = msgs_serializadas[-1]['content']

        payload_data: Dict[str, Any] = {
            'skill': skill,
            'cooperativa_id': str(cooperativa_id) if cooperativa_id else None,
            'empresa_id': str(empresa_id) if empresa_id else None,
            'user_id': user_id,
            'messages': msgs_serializadas,
        }
        if callbacks is not None:
            payload_data['callbacks'] = callbacks

        task_id = str(uuid4())
        return JSONRPCRequest(
            id=str(uuid4()),
            method='tasks/send',
            params={
                'id': task_id,
                'sessionId': session_id,
                'message': A2AMessage(
                    role='user',
                    parts=[
                        DataPart(data=payload_data),
                        TextPart(
                            text=ultima_pergunta
                            or f"Executar skill '{skill}' no serviço A2A {servico}"
                        ),
                    ],
                ).model_dump(),
                'metadata': {
                    'protocol': 'google-a2a',
                    'source': 'host_router_graph',
                    'target_service': f'{servico}_service',
                    'skill': skill,
                },
            },
        )

    def send_task(
        self,
        skill: str,
        cooperativa_id: Optional[str] = None,
        empresa_id: Optional[str] = None,
        messages: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        servico: Optional[ServicoA2A] = None,
        callbacks: Optional[Any] = None,
    ) -> JSONRPCResponse:
        """Envia sincronicamente `tasks/send` ao endpoint A2A de Cooperativa ou Empresa."""
        servico_alvo: ServicoA2A = servico or self.resolver_servico_por_skill(skill)
        rpc_req = self._montar_rpc_request(
            skill=skill,
            servico=servico_alvo,
            cooperativa_id=cooperativa_id,
            empresa_id=empresa_id,
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            callbacks=callbacks,
        )

        with medir_span(f'a2a_send_task_{skill}', categoria='a2a'):
            url_remota = self.cooperativa_url if servico_alvo == 'cooperativa' else self.empresa_url
            if url_remota:
                payload_http = rpc_req.model_dump()
                # Callbacks em memória não são serializáveis via JSON HTTP
                for part in payload_http['params']['message']['parts']:
                    if part.get('type') == 'data' and 'callbacks' in part.get('data', {}):
                        part['data'].pop('callbacks', None)
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url_remota, json=payload_http)
                    resp.raise_for_status()
                    return JSONRPCResponse.model_validate(resp.json())

            servidor_local = (
                self.cooperativa_server if servico_alvo == 'cooperativa' else self.empresa_server
            )
            if servidor_local is None:
                raise RuntimeError(f"Servidor A2A '{servico_alvo}' não configurado no A2AClient.")
            return servidor_local.handle_jsonrpc(rpc_req)

    async def asend_task(
        self,
        skill: str,
        cooperativa_id: Optional[str] = None,
        empresa_id: Optional[str] = None,
        messages: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        servico: Optional[ServicoA2A] = None,
        callbacks: Optional[Any] = None,
    ) -> JSONRPCResponse:
        """Envia assincronamente `tasks/send` ao endpoint A2A de Cooperativa ou Empresa."""
        servico_alvo: ServicoA2A = servico or self.resolver_servico_por_skill(skill)
        rpc_req = self._montar_rpc_request(
            skill=skill,
            servico=servico_alvo,
            cooperativa_id=cooperativa_id,
            empresa_id=empresa_id,
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            callbacks=callbacks,
        )

        with medir_span(f'a2a_asend_task_{skill}', categoria='a2a'):
            url_remota = self.cooperativa_url if servico_alvo == 'cooperativa' else self.empresa_url
            if url_remota:
                payload_http = rpc_req.model_dump()
                for part in payload_http['params']['message']['parts']:
                    if part.get('type') == 'data' and 'callbacks' in part.get('data', {}):
                        part['data'].pop('callbacks', None)
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url_remota, json=payload_http)
                    resp.raise_for_status()
                    return JSONRPCResponse.model_validate(resp.json())

            servidor_local = (
                self.cooperativa_server if servico_alvo == 'cooperativa' else self.empresa_server
            )
            if servidor_local is None:
                raise RuntimeError(f"Servidor A2A '{servico_alvo}' não configurado no A2AClient.")
            return await servidor_local.ahandle_jsonrpc(rpc_req)

    @staticmethod
    def extrair_dados_artifact(rpc_response: JSONRPCResponse) -> Dict[str, Any]:
        """Extrai o dicionário de dados do primeiro Artifact da resposta A2A."""
        if rpc_response.error is not None:
            return {
                'status': 'error',
                'code': rpc_response.error.code,
                'message': rpc_response.error.message,
            }
        task: Optional[Task] = rpc_response.result
        if task is None:
            return {'status': 'error', 'message': 'Resposta A2A sem Task.'}

        if task.status.state == 'failed':
            msg_erro = ''
            if task.status.message and task.status.message.parts:
                primeiro = task.status.message.parts[0]
                msg_erro = getattr(primeiro, 'text', '')
            return {'status': 'error', 'task_id': task.id, 'message': msg_erro or 'Task A2A falhou.'}

        for artifact in task.artifacts:
            for part in artifact.parts:
                if isinstance(part, DataPart):
                    return {
                        'status': 'ok',
                        'task_id': task.id,
                        'task_state': task.status.state,
                        **part.data,
                    }

        return {
            'status': 'ok',
            'task_id': task.id,
            'task_state': task.status.state,
        }

    @classmethod
    def extrair_estado_grafo(cls, rpc_response: JSONRPCResponse) -> Dict[str, Any]:
        """Converte o Artifact retornado pelo servidor A2A na atualização de GraphState
        esperada pelo Orquestrador do Grafo Principal."""
        dados = cls.extrair_dados_artifact(rpc_response)
        if dados.get('status') != 'ok':
            erro_msg = dados.get('message') or 'Falha ao consultar serviço especialista via A2A.'
            return {
                'messages': [{'role': 'assistant', 'content': erro_msg}],
                'agentes_chamados': ['a2a_error'],
                'especialista_outputs': {
                    'a2a_error': EspecialistaOutput(conteudo=erro_msg, fonte='a2a_error')
                },
            }

        nome_agente = str(dados.get('nome_agente') or dados.get('skill') or 'especialista')
        conteudo = str(dados.get('conteudo') or '')
        agentes_chamados = list(dados.get('agentes_chamados') or [nome_agente])

        outputs_raw = dados.get('especialista_outputs') or {}
        especialista_outputs: Dict[str, EspecialistaOutput] = {}
        for chave, info in outputs_raw.items():
            if isinstance(info, EspecialistaOutput):
                especialista_outputs[chave] = info
            elif isinstance(info, dict):
                especialista_outputs[chave] = EspecialistaOutput(
                    conteudo=str(info.get('conteudo') or ''),
                    fonte=str(info.get('fonte') or chave),
                )
        if not especialista_outputs and conteudo:
            especialista_outputs[nome_agente] = EspecialistaOutput(
                conteudo=conteudo,
                fonte=nome_agente,
            )

        return {
            'messages': [{'role': 'assistant', 'content': conteudo}],
            'agentes_chamados': agentes_chamados,
            'especialista_outputs': especialista_outputs,
        }
