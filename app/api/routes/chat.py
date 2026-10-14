from app.api.schemas.chat_request import ChatRequest
from app.api.schemas.chat_response import ChatResponse
from app.guardrails.pii import anonimizar
from app.graph.state import RESET_OUTPUTS

import asyncio
from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage


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

@router.post('/chat', response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    container = request.app.state.container
    mongo_memory = container.memory_stores['mongo_memory']
    obs = container.observabilidade

    turno_id = obs.abrir_turno(session_id=body.session_id)
    status = "erro"
    rota = "—"

    try:
        pergunta, mapa_pii = anonimizar(body.pergunta)

        perfil_ctx = await asyncio.to_thread(
            container.repositories['perfil_repository'].resolve_perfil, body.perfil_id
        )

        await asyncio.to_thread(
            mongo_memory.preparar_turno,
            pergunta, perfil_ctx, body.session_id, body.user_id,
        )

        config = {'configurable': {
            'thread_id': body.session_id,
            'user_id':   body.user_id,
            'perfil_id': body.perfil_id,
            'mapa_pii':  mapa_pii,
        }}

        estado_anterior = await container.graph.aget_state(config)
        qtd_anteriores = len((estado_anterior.values or {}).get('agentes_chamados', []))

        resultado = await container.graph.ainvoke(
            {
                'messages': [HumanMessage(content=pergunta)],
                'perfil_ctx': perfil_ctx,
                'especialista_outputs': {RESET_OUTPUTS: True},
            },
            config=config,
        )

        passou_guardrail = resultado.get('passou_guardrail', True)
        if not passou_guardrail:
            status = "bloqueado"
            rota = "bloqueio_guardrail"
        else:
            status = "ok"
            rotas = resultado.get('proximos_agentes') or []
            if rotas:
                rota = ", ".join(rotas)
            else:
                agentes_totais = resultado.get('agentes_chamados', [])
                agentes_turno_temp = agentes_totais[qtd_anteriores:]
                rota = agentes_turno_temp[-1] if agentes_turno_temp else "fim"

        resposta = _obter_texto_mensagem(resultado['messages'][-1])
        agentes_turno = resultado.get('agentes_chamados', [])[qtd_anteriores:]

        await asyncio.to_thread(
            mongo_memory.registrar_resposta,
            resposta, body.session_id, agentes_turno,
        )

        return ChatResponse(resposta=resposta, agentes_chamados=agentes_turno)
    except Exception as e:
        status = "erro"
        obs.error(f"Falha na execução do chat: {e}", modulo="chat", exc=e)
        raise
    finally:
        obs.fechar_turno(turno_id, session_id=body.session_id, rota=rota, status=status)