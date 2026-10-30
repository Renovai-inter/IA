import asyncio
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from langchain_core.messages import HumanMessage

from app.api.schemas.chat_request import ChatRequest
from app.api.schemas.chat_response import ChatResponse
from app.core.observabilidade import medir_span
from app.graph.state import RESET_AGENTES, RESET_OUTPUTS
from app.guardrails.pii import anonimizar
from app.repository.postgresql.perfil_repository import PerfilContext


router = APIRouter(tags=['chat'])


def _obter_texto_mensagem(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return ''.join(
        bloco.get('text', '')
        for bloco in content
        if isinstance(bloco, dict) and bloco.get('type') == 'text'
    )


def _finalizar_telemetria_redis(
    redis_realtime,
    turno_id: str,
    session_id: str,
    rota: str,
    status: str,
    latencia_ms: int,
    user_id: str | None,
    entidade_id: str | None,
    tipo_fluxo: str,
    rotas_acionadas: list[str],
) -> None:
    if redis_realtime is None:
        return
    redis_realtime.finalizar_turno_fila(
        turno_id=turno_id,
        session_id=session_id,
        rota=rota,
        status=status,
        latencia_ms=latencia_ms,
        user_id=user_id,
        entidade_id=entidade_id,
        tipo_fluxo=tipo_fluxo,
    )
    rotas_ranking = rotas_acionadas if rotas_acionadas else ([rota] if rota and rota != '—' else [])
    redis_realtime.registrar_interacao_ranking(
        entidade_id=entidade_id,
        tipo_fluxo=tipo_fluxo,
        rotas=rotas_ranking,
        user_id=user_id or session_id,
    )


@router.post('/chat', response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest, background_tasks: BackgroundTasks):
    container = request.app.state.container
    mongo_memory = container.memory_stores['mongo_memory']
    obs = container.observabilidade
    redis_realtime = getattr(container, 'redis_realtime', None)

    turno_id = obs.abrir_turno(session_id=body.session_id)
    t0_turno = time.perf_counter()
    status = 'erro'
    rota = '—'
    rotas_acionadas: list[str] = []
    enfileirado_no_redis = False

    try:
        # 1. Rate Limiting (Sliding Window Log no Redis via ZSET)
        if redis_realtime is not None:
            identificador_rl = body.user_id or body.entidade_id or body.session_id
            with medir_span('redis_rate_limit', categoria='redis'):
                rl = await asyncio.to_thread(
                    redis_realtime.verificar_rate_limit,
                    identificador_rl,
                )
            if not rl.permitido:
                status = 'bloqueado'
                rota = 'rate_limit'
                obs.warning(
                    f"Rate limit excedido para '{identificador_rl}' "
                    f"({rl.usado_na_janela}/{rl.limite} em {rl.janela_segundos}s).",
                    modulo='redis',
                )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Limite de requisições excedido ({rl.limite} mensagens a cada "
                        f"{rl.janela_segundos}s). Aguarde {rl.retry_after_segundos}s."
                    ),
                    headers={
                        'Retry-After': str(rl.retry_after_segundos),
                        'X-RateLimit-Limit': str(rl.limite),
                        'X-RateLimit-Remaining': '0',
                    },
                )

        pergunta, mapa_pii = anonimizar(body.pergunta)

        perfil_ctx = PerfilContext.from_request(
            cooperativa_id=body.cooperativa_id,
            empresa_id=body.empresa_id,
        )

        tarefas_prep = [
            asyncio.to_thread(
                mongo_memory.preparar_turno,
                pergunta, perfil_ctx, body.session_id, body.user_id,
            )
        ]
        if redis_realtime is not None:
            def _enfileirar_sync():
                with medir_span('redis_fila_enqueue', categoria='redis'):
                    redis_realtime.enfileirar_turno(
                        turno_id=turno_id,
                        session_id=body.session_id,
                        user_id=body.user_id,
                        entidade_id=body.entidade_id,
                        tipo_fluxo=body.tipo_fluxo,
                        pergunta=body.pergunta,
                    )
            tarefas_prep.append(asyncio.to_thread(_enfileirar_sync))
            enfileirado_no_redis = True

        await asyncio.gather(*tarefas_prep)

        config = {'configurable': {
            'thread_id':      body.session_id,
            'user_id':        body.user_id,
            'cooperativa_id': body.cooperativa_id,
            'empresa_id':     body.empresa_id,
            'perfil_id':      body.entidade_id,
            'mapa_pii':       mapa_pii,
        }}

        resultado = await container.graph.ainvoke(
            {
                'messages': [HumanMessage(content=pergunta)],
                'cooperativa_id': perfil_ctx.cooperativa_id,
                'empresa_id': perfil_ctx.empresa_id,
                'perfil_ctx': perfil_ctx,
                'especialista_outputs': {RESET_OUTPUTS: True},
                'agentes_chamados': [RESET_AGENTES],
            },
            config=config,
        )

        agentes_turno = resultado.get('agentes_chamados', [])
        rotas_acionadas = list(resultado.get('proximos_agentes') or [])

        passou_guardrail = resultado.get('passou_guardrail', True)
        passou_juiz = resultado.get('passou_juiz', True)
        if not passou_guardrail:
            status = 'bloqueado'
            rota = "bloqueio_guardrail"
        elif not passou_juiz:
            status = 'recusado_juiz'
            rota = 'bloqueio_juiz'
        else:
            status = 'ok'
            if rotas_acionadas:
                rota = ', '.join(rotas_acionadas)
            else:
                rota = agentes_turno[-1] if agentes_turno else 'fim'

        resposta = _obter_texto_mensagem(resultado['messages'][-1])
        latencia_ms = int((time.perf_counter() - t0_turno) * 1000)

        # Gravação assíncrona do histórico no MongoDB Atlas e atualização da Fila/Ranking no Redis
        # via BackgroundTasks, liberando o envio imediato da resposta HTTP ao cliente.
        background_tasks.add_task(
            mongo_memory.registrar_resposta,
            resposta, body.session_id, agentes_turno,
        )
        if redis_realtime is not None and enfileirado_no_redis:
            enfileirado_no_redis = False
            background_tasks.add_task(
                _finalizar_telemetria_redis,
                redis_realtime,
                turno_id,
                body.session_id,
                rota,
                status,
                latencia_ms,
                body.user_id,
                body.entidade_id,
                body.tipo_fluxo,
                rotas_acionadas,
            )

        return ChatResponse(resposta=resposta, agentes_chamados=agentes_turno)
    except HTTPException:
        raise
    except Exception as e:
        status = 'erro'
        obs.error(f"Falha na execução do chat: {e}", modulo='chat', exc=e)
        raise
    finally:
        if redis_realtime is not None and enfileirado_no_redis:
            lat_erro_ms = int((time.perf_counter() - t0_turno) * 1000)
            try:
                _finalizar_telemetria_redis(
                    redis_realtime,
                    turno_id,
                    body.session_id,
                    rota,
                    status,
                    lat_erro_ms,
                    body.user_id,
                    body.entidade_id,
                    body.tipo_fluxo,
                    rotas_acionadas,
                )
            except Exception:
                pass
        obs.fechar_turno(turno_id, session_id=body.session_id, rota=rota, status=status)
