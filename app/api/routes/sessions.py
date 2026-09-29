from fastapi import APIRouter, Request

from app.api.schemas.session_response import SessionResponse
from app.memory.mongo_memory import MongoMemory


router = APIRouter(prefix='/sessions', tags=['sessions'])

@router.post('/{session_id}/encerrar', response_model=SessionResponse)
def encerrar(request: Request, session_id) -> SessionResponse:
    container = request.app.state.container
    mongo_memory = container.memory_stores['mongo_memory']
    qdrant_memory = container.memory_stores['qdrant_memory']

    sessao_encerrada = mongo_memory.encerrar_sessao(session_id)

    if sessao_encerrada.resumo:
        qdrant_memory.atualizar_memoria(sessao_encerrada)

    return SessionResponse(
        session_id=session_id,
        resumo=sessao_encerrada.resumo,
        agentes_chamados=sessao_encerrada.agentes_chamados
    )