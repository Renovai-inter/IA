"""Servidor MCP (Model Context Protocol — JSON-RPC 2.0) do ecossistema Renovaí.

Padroniza a descoberta e invocação de Tools (`tools/list`, `tools/call`),
Resources (`resources/list`, `resources/read`) e Prompts (`prompts/list`, `prompts/get`)
para os agentes especialistas do grafo e clientes MCP externos.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import date, datetime
from decimal import Decimal
import functools
import inspect
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Query, Request
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from app.core.observabilidade import medir_span
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
from app.repository.postgresql.perfil_repository import _to_uuid

log = logging.getLogger('renovai.mcp.server')

router = APIRouter(tags=['mcp'])

_DOMINIO_POR_TOOLKIT: Dict[str, str] = {
    'material_estoque_toolkit': 'COOPERATIVA',
    'coleta_toolkit': 'COOPERATIVA',
    'financeiro_toolkit': 'COOPERATIVA',
    'recomendacao_toolkit': 'EMPRESA',
    'faq_toolkit': 'TRANSVERSAL',
    'memoria_toolkit': 'TRANSVERSAL',
}

_REPOS_POR_TOOLKIT: Dict[str, List[str]] = {
    'material_estoque_toolkit': [
        'material_repository',
        'estoque_repository',
        'movimentacao_estoque_repository',
    ],
    'coleta_toolkit': [
        'coleta_repository',
        'triagem_repository',
        'funcionario_repository',
        'equipe_repository',
        'material_repository',
    ],
    'financeiro_toolkit': [
        'rateio_repository',
        'rateio_funcionario_repository',
        'despesa_repository',
        'funcionario_repository',
    ],
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (UUID, Decimal)):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    return str(obj)


def _extrair_input_schema(tool: StructuredTool) -> Dict[str, Any]:
    args_schema = getattr(tool, 'args_schema', None)
    if args_schema is not None and hasattr(args_schema, 'model_json_schema'):
        try:
            schema = dict(args_schema.model_json_schema())
            schema.setdefault('type', 'object')
            return schema
        except Exception:
            pass
    return {'type': 'object', 'properties': {}}


class MCPServer:
    """Servidor Model Context Protocol (MCP) integrado ao Container do Renovaí."""

    def __init__(
        self,
        toolkits: Optional[Dict[str, Any]] = None,
        repositories: Optional[Dict[str, Any]] = None,
        observabilidade: Optional[Any] = None,
        redis_realtime: Optional[Any] = None,
        pnrs_path: Optional[str] = None,
    ):
        self.toolkits = toolkits or {}
        self.repositories = repositories or {}
        self.observabilidade = observabilidade
        self.redis_realtime = redis_realtime
        self.pnrs_path = pnrs_path

        self._lock = threading.Lock()
        self._tools_map: Dict[str, StructuredTool] = {}
        self._specs_map: Dict[str, MCPToolSpec] = {}
        self._stats_por_tool: Dict[str, Dict[str, Any]] = {}
        self._total_chamadas: int = 0
        self._total_erros: int = 0
        self._total_resources_lidos: int = 0

        for tk_name, tk_instance in self.toolkits.items():
            if hasattr(tk_instance, 'get_tools'):
                self.registrar_toolkit(tk_name, tk_instance.get_tools())

    def registrar_toolkit(self, toolkit_name: str, tools: List[StructuredTool]) -> None:
        dominio = _DOMINIO_POR_TOOLKIT.get(toolkit_name, 'TRANSVERSAL')
        for tool in tools:
            spec = MCPToolSpec(
                name=tool.name,
                description=(tool.description or '').strip(),
                inputSchema=_extrair_input_schema(tool),
                toolkit=toolkit_name,
                domain=dominio,
            )
            self._tools_map[tool.name] = tool
            self._specs_map[tool.name] = spec
            with self._lock:
                self._stats_por_tool.setdefault(
                    tool.name,
                    {
                        'tool': tool.name,
                        'toolkit': toolkit_name,
                        'dominio': dominio,
                        'chamadas': 0,
                        'erros': 0,
                        'latencia_total_ms': 0,
                    },
                )

    def registrar_telemetria_tool(self, tool_name: str, duracao_ms: int, erro: bool = False) -> None:
        with self._lock:
            self._total_chamadas += 1
            if erro:
                self._total_erros += 1
            st = self._stats_por_tool.get(tool_name)
            if st is not None:
                st['chamadas'] += 1
                if erro:
                    st['erros'] += 1
                st['latencia_total_ms'] += max(0, int(duracao_ms))

    def wrap_toolkit_tools(
        self,
        toolkit_name: str,
        tools: List[StructuredTool],
    ) -> List[StructuredTool]:
        """Registra as tools no servidor MCP e devolve StructuredTools instrumentadas via protocolo MCP."""
        self.registrar_toolkit(toolkit_name, tools)
        wrapped: List[StructuredTool] = []

        for tool in tools:
            orig_func = tool.func
            if orig_func is None:
                wrapped.append(tool)
                continue

            tool_name = tool.name

            @functools.wraps(orig_func)
            def _mcp_bridged_func(*args, __fn=orig_func, __tname=tool_name, **kwargs):
                t0 = time.perf_counter()
                erro = False
                try:
                    with medir_span(f'mcp:{__tname}', categoria='mcp'):
                        res = __fn(*args, **kwargs)
                    if isinstance(res, dict) and res.get('status') == 'error':
                        erro = False
                    return res
                except Exception:
                    erro = True
                    raise
                finally:
                    dur_ms = int((time.perf_counter() - t0) * 1000)
                    self.registrar_telemetria_tool(__tname, dur_ms, erro=erro)

            novo_tool = StructuredTool.from_function(
                func=_mcp_bridged_func,
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema,
            )
            self._tools_map[tool.name] = novo_tool
            wrapped.append(novo_tool)

        return wrapped

    # --------------------------------------------------------------------------
    # Listagem de Tools, Resources e Prompts
    # --------------------------------------------------------------------------
    def listar_tools(self, domain: Optional[str] = None) -> List[MCPToolSpec]:
        specs = list(self._specs_map.values())
        if domain:
            dom_upper = domain.strip().upper()
            specs = [s for s in specs if (s.domain or '').upper() == dom_upper]
        return specs

    def listar_resources(self) -> List[MCPResourceSpec]:
        return [
            MCPResourceSpec(
                uri='renovai://normas/pnrs',
                name='Política Nacional de Resíduos Sólidos (Lei 12.305/2010)',
                description='Texto normativo de referência da PNRS utilizado pelos agentes de conformidade e FAQ.',
                mimeType='text/plain',
            ),
            MCPResourceSpec(
                uri='renovai://telemetria/sre',
                name='Painel de Observabilidade SRE, SLO e FinOps (ROI)',
                description='Métricas em tempo real de latência P95, custo por resolução, projeção de 100/1.000 usuários e ROI.',
                mimeType='application/json',
            ),
            MCPResourceSpec(
                uri='renovai://telemetria/redis',
                name='Fila de Processamento e Rankings em Tempo Real (Redis)',
                description='Estado atual da fila de turnos ativos/recentes, bloqueios de Rate Limit e leaderboards ZSET.',
                mimeType='application/json',
            ),
            MCPResourceSpec(
                uri='renovai://cooperativa/{cooperativa_id}/estoque',
                name='Snapshot Operacional de Estoque e Materiais da Cooperativa',
                description='Catálogo de materiais recicláveis, saldos em estoque (kg) e histórico de movimentações.',
                mimeType='application/json',
            ),
            MCPResourceSpec(
                uri='renovai://cooperativa/{cooperativa_id}/financeiro',
                name='Snapshot Financeiro de Rateios e Despesas da Cooperativa',
                description='Registros de rateios cooperados, pagamentos por funcionário e despesas operacionais.',
                mimeType='application/json',
            ),
            MCPResourceSpec(
                uri='renovai://cooperativa/{cooperativa_id}/coletas',
                name='Snapshot de Coletas, Triagens e Equipes da Cooperativa',
                description='Agendamentos de coleta, lotes triados, equipes e cooperados vinculados.',
                mimeType='application/json',
            ),
        ]

    def listar_prompts(self) -> List[MCPPromptSpec]:
        return [
            MCPPromptSpec(
                name='diagnostico_estoque_cooperativa',
                description='Gera um prompt estruturado para análise crítica de giro de estoque e gargalos de triagem.',
                arguments=[
                    MCPPromptArgument(
                        name='cooperativa_id',
                        description='UUID da cooperativa a ser auditada.',
                        required=True,
                    ),
                    MCPPromptArgument(
                        name='material_foco',
                        description='Material reciclável prioritário (ex.: PET, Papelão, Alumínio).',
                        required=False,
                    ),
                ],
            ),
            MCPPromptSpec(
                name='auditoria_rateio_financeiro',
                description='Gera um prompt estruturado para conciliação de despesas, receitas e rateio entre cooperados.',
                arguments=[
                    MCPPromptArgument(
                        name='cooperativa_id',
                        description='UUID da cooperativa.',
                        required=True,
                    ),
                    MCPPromptArgument(
                        name='periodo',
                        description='Período de referência (ex.: último mês, trimestre atual).',
                        required=False,
                    ),
                ],
            ),
            MCPPromptSpec(
                name='parecer_conformidade_pnrs',
                description='Gera um prompt para enquadramento de operações de reciclagem e logística reversa na Lei 12.305/2010.',
                arguments=[
                    MCPPromptArgument(
                        name='tema',
                        description='Tema regulatório ou dúvida operacional sobre resíduos sólidos.',
                        required=True,
                    ),
                ],
            ),
        ]

    # --------------------------------------------------------------------------
    # Execução de Tools via MCP (`tools/call`)
    # --------------------------------------------------------------------------
    def _preparar_config_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> tuple[Dict[str, Any], RunnableConfig]:
        args_limpos = dict(arguments or {})
        meta_dict = dict(meta or {})

        # Extrai metadados de contexto se passados junto aos argumentos da chamada MCP externa
        coop_raw = (
            args_limpos.pop('_cooperativa_id', None)
            or args_limpos.pop('cooperativa_id', None)
            or meta_dict.get('cooperativa_id')
            or os.environ.get('DEFAULT_COOPERATIVA_ID')
        )
        emp_raw = (
            args_limpos.pop('_empresa_id', None)
            or args_limpos.pop('empresa_id', None)
            or meta_dict.get('empresa_id')
            or os.environ.get('DEFAULT_EMPRESA_ID')
        )
        session_id = args_limpos.pop('_session_id', None) or meta_dict.get('session_id') or 'mcp_session'
        user_id = args_limpos.pop('_user_id', None) or meta_dict.get('user_id') or 'mcp_client'

        base_cfg = dict(config or {})
        configurable = dict(base_cfg.get('configurable') or {})

        coop_uuid = _to_uuid(coop_raw or configurable.get('cooperativa_id'))
        emp_uuid = _to_uuid(emp_raw or configurable.get('empresa_id'))

        if coop_uuid is None and emp_uuid is None:
            try:
                pg_repo = self.repositories.get('material_repository') or self.repositories.get('estoque_repository')
                if pg_repo and hasattr(pg_repo, '_db') and pg_repo._db:
                    with pg_repo._db.connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute('SELECT cooperativa_id FROM cooperativas LIMIT 1')
                            row = cur.fetchone()
                            if row:
                                val = row.get('cooperativa_id') if isinstance(row, dict) else row[0]
                                coop_uuid = _to_uuid(val)
            except Exception:
                pass

        if coop_uuid is not None:
            configurable['cooperativa_id'] = str(coop_uuid)
        if emp_uuid is not None:
            configurable['empresa_id'] = str(emp_uuid)
        configurable.setdefault('thread_id', session_id)
        configurable.setdefault('user_id', user_id)

        # Se não houver snapshots pré-carregados e a tool pertencer a um toolkit de cooperativa, carrega sob demanda
        snapshots = dict(configurable.get('snapshots') or {})
        spec = self._specs_map.get(tool_name)
        tk_name = spec.toolkit if spec else None
        repos_necessarios = _REPOS_POR_TOOLKIT.get(tk_name or '', [])

        if repos_necessarios and coop_uuid is not None and not snapshots:
            for repo_nome in repos_necessarios:
                repo = self.repositories.get(repo_nome)
                if repo is not None:
                    snap_key = repo_nome.replace('repository', 'snapshot')
                    getter = getattr(repo, 'get_snapshot_cached', getattr(repo, 'get_snapshot', None))
                    if callable(getter):
                        try:
                            snapshots[snap_key] = getter(coop_uuid)
                        except Exception as exc:
                            log.warning(f'[mcp] Falha ao carregar {snap_key} para {tool_name}: {exc}')
            configurable['snapshots'] = snapshots

        base_cfg['configurable'] = configurable
        return args_limpos, base_cfg  # type: ignore[return-value]

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> MCPCallToolResult:
        """Executa uma tool registrada pelo nome seguindo o contrato MCP `tools/call`."""
        tool = self._tools_map.get(name)
        if tool is None:
            raise KeyError(f"Tool MCP '{name}' não encontrada.")

        args_limpos, run_cfg = self._preparar_config_tool(name, arguments or {}, meta=meta, config=config)

        try:
            resultado_bruto = tool.invoke(args_limpos, config=run_cfg)
            is_err = isinstance(resultado_bruto, dict) and resultado_bruto.get('status') == 'error'
            texto_serializado = (
                resultado_bruto
                if isinstance(resultado_bruto, str)
                else json.dumps(resultado_bruto, ensure_ascii=False, default=_json_default)
            )
            return MCPCallToolResult(
                content=[MCPContentBlock(type='text', text=texto_serializado)],
                isError=is_err,
                structuredContent=resultado_bruto,
            )
        except Exception as exc:
            log.warning(f"[mcp] Erro ao executar tool '{name}': {exc}")
            erro_payload = {'status': 'error', 'message': str(exc)}
            return MCPCallToolResult(
                content=[MCPContentBlock(type='text', text=json.dumps(erro_payload, ensure_ascii=False))],
                isError=True,
                structuredContent=erro_payload,
            )

    async def acall_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> MCPCallToolResult:
        return await asyncio.to_thread(self.call_tool, name, arguments, meta, config)

    # --------------------------------------------------------------------------
    # Leitura de Resources (`resources/read`) e Prompts (`prompts/get`)
    # --------------------------------------------------------------------------
    def read_resource(self, uri: str) -> MCPReadResourceResult:
        uri_clean = (uri or '').strip()
        with self._lock:
            self._total_resources_lidos += 1

        with medir_span(f'mcp:resource:{uri_clean[:40]}', categoria='mcp'):
            if uri_clean == 'renovai://normas/pnrs':
                texto_pnrs = 'Política Nacional de Resíduos Sólidos (Lei nº 12.305/2010).'
                if self.pnrs_path and Path(self.pnrs_path).exists():
                    try:
                        texto_pnrs = Path(self.pnrs_path).read_text(encoding='utf-8')[:8000]
                    except Exception:
                        pass
                return MCPReadResourceResult(
                    contents=[
                        MCPResourceContent(
                            uri=uri_clean,
                            mimeType='text/plain',
                            text=texto_pnrs,
                        )
                    ]
                )

            if uri_clean == 'renovai://telemetria/sre':
                dados_sre = (
                    self.observabilidade.painel()
                    if self.observabilidade and hasattr(self.observabilidade, 'painel')
                    else {'status': 'indisponivel'}
                )
                return MCPReadResourceResult(
                    contents=[
                        MCPResourceContent(
                            uri=uri_clean,
                            mimeType='application/json',
                            text=json.dumps(dados_sre, ensure_ascii=False, default=_json_default),
                        )
                    ]
                )

            if uri_clean == 'renovai://telemetria/redis':
                dados_redis = (
                    self.redis_realtime.obter_painel_realtime()
                    if self.redis_realtime and hasattr(self.redis_realtime, 'obter_painel_realtime')
                    else {'status': 'indisponivel'}
                )
                return MCPReadResourceResult(
                    contents=[
                        MCPResourceContent(
                            uri=uri_clean,
                            mimeType='application/json',
                            text=json.dumps(dados_redis, ensure_ascii=False, default=_json_default),
                        )
                    ]
                )

            if uri_clean.startswith('renovai://cooperativa/'):
                partes = uri_clean.removeprefix('renovai://cooperativa/').split('/')
                if len(partes) == 2:
                    coop_id_str, dominio_res = partes[0], partes[1]
                    coop_uuid = _to_uuid(coop_id_str)
                    if coop_uuid is None:
                        raise ValueError(f"UUID de cooperativa inválido no resource URI: '{coop_id_str}'")

                    mapa_dominio_tk = {
                        'estoque': 'material_estoque_toolkit',
                        'financeiro': 'financeiro_toolkit',
                        'coletas': 'coleta_toolkit',
                    }
                    tk_alvo = mapa_dominio_tk.get(dominio_res)
                    if tk_alvo:
                        payload_snap: Dict[str, Any] = {'cooperativa_id': str(coop_uuid), 'dominio': dominio_res}
                        for repo_nome in _REPOS_POR_TOOLKIT.get(tk_alvo, []):
                            repo = self.repositories.get(repo_nome)
                            if repo is not None:
                                getter = getattr(repo, 'get_snapshot_cached', getattr(repo, 'get_snapshot', None))
                                if callable(getter):
                                    payload_snap[repo_nome.replace('_repository', '')] = getter(coop_uuid)
                        return MCPReadResourceResult(
                            contents=[
                                MCPResourceContent(
                                    uri=uri_clean,
                                    mimeType='application/json',
                                    text=json.dumps(payload_snap, ensure_ascii=False, default=_json_default),
                                )
                            ]
                        )

        raise KeyError(f"Resource MCP '{uri_clean}' não encontrado.")

    def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPGetPromptResult:
        args = arguments or {}
        if name == 'diagnostico_estoque_cooperativa':
            coop_id = args.get('cooperativa_id', 'não informada')
            mat = args.get('material_foco', 'todos os materiais recicláveis')
            texto = (
                f'Realize um diagnóstico operacional completo do estoque da cooperativa {coop_id} '
                f'com foco em {mat}. Consulte as tools MCP `consultar_estoque` e `buscar_estoque_historico`, '
                'identificando saldos críticos, giro de entradas/saídas e sugestões de escoamento.'
            )
            return MCPGetPromptResult(
                description='Diagnóstico de estoque e movimentações da cooperativa.',
                messages=[MCPPromptMessage(role='user', content=MCPContentBlock(text=texto))],
            )

        if name == 'auditoria_rateio_financeiro':
            coop_id = args.get('cooperativa_id', 'não informada')
            periodo = args.get('periodo', 'período recente')
            texto = (
                f'Execute uma auditoria financeira para a cooperativa {coop_id} referente ao {periodo}. '
                'Acione as tools MCP `consultar_rateios`, `consultar_pagamentos_funcionarios` e `consultar_despesas`, '
                'cruzando o saldo líquido rateado com os custos operacionais.'
            )
            return MCPGetPromptResult(
                description='Auditoria de rateio cooperado e despesas operacionais.',
                messages=[MCPPromptMessage(role='user', content=MCPContentBlock(text=texto))],
            )

        if name == 'parecer_conformidade_pnrs':
            tema = args.get('tema', 'responsabilidade compartilhada e logística reversa')
            texto = (
                f'Elabore um parecer de conformidade ambiental sobre "{tema}" fundamentado na '
                'Política Nacional de Resíduos Sólidos (Lei nº 12.305/2010), acionando a tool MCP `normas_retriever`.'
            )
            return MCPGetPromptResult(
                description='Parecer de conformidade legal na PNRS (Lei 12.305/2010).',
                messages=[MCPPromptMessage(role='user', content=MCPContentBlock(text=texto))],
            )

        raise KeyError(f"Prompt MCP '{name}' não encontrado.")

    # --------------------------------------------------------------------------
    # Handler JSON-RPC 2.0 (`POST /mcp`)
    # --------------------------------------------------------------------------
    def handle_rpc(self, rpc_req: MCPJSONRPCRequest) -> MCPJSONRPCResponse:
        method = rpc_req.method
        params = rpc_req.params or {}

        try:
            if method == 'initialize':
                init_res = MCPInitializeResult(
                    protocolVersion=MCP_PROTOCOL_VERSION,
                    capabilities=MCPCapabilities(),
                    serverInfo=MCPServerInfo(),
                )
                return MCPJSONRPCResponse(id=rpc_req.id, result=init_res.model_dump())

            if method in ('notifications/initialized', 'ping'):
                return MCPJSONRPCResponse(id=rpc_req.id, result={'status': 'ok'})

            if method == 'tools/list':
                domain = params.get('domain')
                tools_list = [t.model_dump() for t in self.listar_tools(domain=domain)]
                return MCPJSONRPCResponse(id=rpc_req.id, result={'tools': tools_list})

            if method == 'tools/call':
                tool_name = str(params.get('name') or '').strip()
                if not tool_name:
                    return MCPJSONRPCResponse(
                        id=rpc_req.id,
                        error=MCPJSONRPCError(code=-32602, message="Parâmetro obrigatório 'name' ausente em tools/call."),
                    )
                tool_args = params.get('arguments') or {}
                meta = params.get('_meta') or params.get('meta') or {}
                call_res = self.call_tool(tool_name, tool_args, meta=meta)
                return MCPJSONRPCResponse(id=rpc_req.id, result=call_res.model_dump())

            if method == 'resources/list':
                res_list = [r.model_dump() for r in self.listar_resources()]
                return MCPJSONRPCResponse(id=rpc_req.id, result={'resources': res_list})

            if method == 'resources/read':
                uri = str(params.get('uri') or '').strip()
                if not uri:
                    return MCPJSONRPCResponse(
                        id=rpc_req.id,
                        error=MCPJSONRPCError(code=-32602, message="Parâmetro obrigatório 'uri' ausente em resources/read."),
                    )
                read_res = self.read_resource(uri)
                return MCPJSONRPCResponse(id=rpc_req.id, result=read_res.model_dump())

            if method == 'prompts/list':
                prompts_list = [p.model_dump() for p in self.listar_prompts()]
                return MCPJSONRPCResponse(id=rpc_req.id, result={'prompts': prompts_list})

            if method == 'prompts/get':
                prompt_name = str(params.get('name') or '').strip()
                if not prompt_name:
                    return MCPJSONRPCResponse(
                        id=rpc_req.id,
                        error=MCPJSONRPCError(code=-32602, message="Parâmetro obrigatório 'name' ausente em prompts/get."),
                    )
                prompt_args = params.get('arguments') or {}
                prompt_res = self.get_prompt(prompt_name, prompt_args)
                return MCPJSONRPCResponse(id=rpc_req.id, result=prompt_res.model_dump())

            return MCPJSONRPCResponse(
                id=rpc_req.id,
                error=MCPJSONRPCError(code=-32601, message=f"Método MCP desconhecido: '{method}'"),
            )
        except KeyError as exc:
            return MCPJSONRPCResponse(
                id=rpc_req.id,
                error=MCPJSONRPCError(code=-32602, message=str(exc)),
            )
        except Exception as exc:
            log.exception(f'[mcp] Erro ao processar método JSON-RPC {method}')
            return MCPJSONRPCResponse(
                id=rpc_req.id,
                error=MCPJSONRPCError(code=-32603, message=f'Erro interno no servidor MCP: {exc}'),
            )

    async def ahandle_rpc(self, rpc_req: MCPJSONRPCRequest) -> MCPJSONRPCResponse:
        return await asyncio.to_thread(self.handle_rpc, rpc_req)

    # --------------------------------------------------------------------------
    # Telemetria e Painel MCP para `/monitor`
    # --------------------------------------------------------------------------
    def obter_painel_mcp(self) -> Dict[str, Any]:
        with self._lock:
            stats_lista = []
            for st in self._stats_por_tool.values():
                chamadas = st['chamadas']
                media_ms = round(st['latencia_total_ms'] / chamadas, 1) if chamadas > 0 else 0.0
                stats_lista.append(
                    {
                        'tool': st['tool'],
                        'toolkit': st['toolkit'],
                        'dominio': st['dominio'],
                        'chamadas': chamadas,
                        'erros': st['erros'],
                        'latencia_media_ms': media_ms,
                    }
                )
            stats_lista.sort(key=lambda x: (-x['chamadas'], x['tool']))
            total_chamadas = self._total_chamadas
            total_erros = self._total_erros
            total_res = self._total_resources_lidos

        return {
            'protocolo': f'MCP JSON-RPC 2.0 ({MCP_PROTOCOL_VERSION})',
            'servidor': 'renovai-mcp-server',
            'endpoints': {
                'jsonrpc': '/mcp',
                'manifest': '/.well-known/mcp.json',
                'tools': '/mcp/tools',
                'resources': '/mcp/resources',
                'prompts': '/mcp/prompts',
            },
            'resumo': {
                'tools_registradas': len(self._specs_map),
                'resources_registrados': len(self.listar_resources()),
                'prompts_registrados': len(self.listar_prompts()),
                'total_chamadas_tools': total_chamadas,
                'total_erros_tools': total_erros,
                'total_resources_lidos': total_res,
            },
            'por_tool': stats_lista,
        }


def _obter_mcp_server(request: Request) -> MCPServer:
    container = getattr(request.app.state, 'container', None)
    srv = getattr(container, 'mcp_server', None)
    if srv is None:
        srv = MCPServer()
    return srv


@router.get('/.well-known/mcp.json')
@router.get('/mcp/manifest')
async def mcp_manifest(request: Request) -> Dict[str, Any]:
    """Endpoint de descoberta do Servidor MCP do Renovaí."""
    srv = _obter_mcp_server(request)
    painel = srv.obter_painel_mcp()
    return {
        'protocolVersion': MCP_PROTOCOL_VERSION,
        'serverInfo': MCPServerInfo().model_dump(),
        'capabilities': MCPCapabilities().model_dump(),
        'tools': [t.model_dump() for t in srv.listar_tools()],
        'resources': [r.model_dump() for r in srv.listar_resources()],
        'prompts': [p.model_dump() for p in srv.listar_prompts()],
        'telemetry': painel['resumo'],
    }


@router.get('/mcp/tools')
async def mcp_list_tools(
    request: Request,
    domain: Optional[str] = Query(default=None, description='Filtra por domínio: COOPERATIVA, EMPRESA ou TRANSVERSAL'),
) -> Dict[str, Any]:
    """Lista todas as ferramentas MCP registradas com seus JSON Schemas e estatísticas de uso."""
    srv = _obter_mcp_server(request)
    painel = srv.obter_painel_mcp()
    return {
        'protocolVersion': MCP_PROTOCOL_VERSION,
        'count': len(srv.listar_tools(domain=domain)),
        'tools': [t.model_dump() for t in srv.listar_tools(domain=domain)],
        'stats': painel['por_tool'],
    }


@router.get('/mcp/resources')
async def mcp_list_resources(request: Request) -> Dict[str, Any]:
    """Lista os recursos contextuais (`renovai://...`) expostos pelo servidor MCP."""
    srv = _obter_mcp_server(request)
    resources = [r.model_dump() for r in srv.listar_resources()]
    return {
        'protocolVersion': MCP_PROTOCOL_VERSION,
        'count': len(resources),
        'resources': resources,
    }


@router.get('/mcp/prompts')
async def mcp_list_prompts(request: Request) -> Dict[str, Any]:
    """Lista os templates de prompt padronizados expostos pelo servidor MCP."""
    srv = _obter_mcp_server(request)
    prompts = [p.model_dump() for p in srv.listar_prompts()]
    return {
        'protocolVersion': MCP_PROTOCOL_VERSION,
        'count': len(prompts),
        'prompts': prompts,
    }


@router.post('/mcp')
async def mcp_jsonrpc_endpoint(request: Request, rpc_req: MCPJSONRPCRequest) -> MCPJSONRPCResponse:
    """Endpoint oficial JSON-RPC 2.0 do protocolo MCP (`initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`)."""
    srv = _obter_mcp_server(request)
    return await srv.ahandle_rpc(rpc_req)
