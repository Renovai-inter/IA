from fastapi import APIRouter, HTTPException, Request
import asyncio

from app.api.schemas.session_response import SessionResponse


router = APIRouter(prefix='/sessions', tags=['sessions'])

@router.post('/{session_id}/encerrar', response_model=SessionResponse)
async def encerrar(request: Request, session_id: str) -> SessionResponse:
    container = request.app.state.container
    mongo_memory = container.memory_stores['mongo_memory']
    qdrant_memory = container.memory_stores['qdrant_memory']

    sessao_encerrada = await asyncio.to_thread(mongo_memory.encerrar_sessao, session_id)
    if sessao_encerrada is None:
        raise HTTPException(status_code=404, detail='Nenhuma sessão ativa com mensagens encontrada para este session_id.')

    if sessao_encerrada.resumo:
        await asyncio.to_thread(qdrant_memory.atualizar_memoria, sessao_encerrada)

    await container.checkpointer.adelete_thread(session_id)

    return SessionResponse(
        session_id=session_id,
        resumo=sessao_encerrada.resumo,
        agentes_chamados=sessao_encerrada.agentes_chamados,
    )