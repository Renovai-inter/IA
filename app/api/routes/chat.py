from app.api.schemas.chat_request import ChatRequest
from app.api.schemas.chat_response import ChatResponse
from app.guardrails.pii import anonimizar
from app.graph.state import RESET_OUTPUTS, RESET_AGENTES
from app.repository.postgresql.perfil_repository import PerfilContext

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

        perfil_ctx = PerfilContext.from_request(
            cooperativa_id=body.cooperativa_id,
            empresa_id=body.empresa_id,
        )

        await asyncio.to_thread(
            mongo_memory.preparar_turno,
            pergunta, perfil_ctx, body.session_id, body.user_id,
        )

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
                rota = agentes_turno[-1] if agentes_turno else "fim"

        resposta = _obter_texto_mensagem(resultado['messages'][-1])

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
