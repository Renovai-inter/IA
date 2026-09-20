from fastapi import APIRouter

from app.api.schemas.session_response import SessionResponse
from app.memory.mongo_memory import MongoMemory


router = APIRouter(prefix='/sessions', tags=['sessions'])
mongo_memory = MongoMemory()

@router.post('/{session_id}/encerrar', response_model=SessionResponse)
def encerrar(session_id) -> SessionResponse:
    sessao_encerrada = mongo_memory.encerrar_sessao(session_id)

    return SessionResponse(
        session_id=session_id,
        resumo=sessao_encerrada.resumo,
        agentes_chamados=sessao_encerrada.agentes_chamados
    )