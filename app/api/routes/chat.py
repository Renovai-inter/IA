from app.api.schemas.chat_request import ChatRequest
from app.api.schemas.chat_response import ChatResponse
from app.core.config import Settings


from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage

from fastapi import APIRouter


settings = Settings()
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
    perfil_ctx = container.repositories['perfil_repository'].resolve_perfil(body.perfil_id)

    resultado = await container.graph.ainvoke(
        {'messages': [HumanMessage(content=body.pergunta)], 'perfil_ctx': perfil_ctx},
        config={'configurable': {'thread_id': body.session_id, 'user_id': body.user_id, 'perfil_id': body.perfil_id}},
    )

    return ChatResponse(resposta=_obter_texto_mensagem(resultado['messages'][-1]))
    # return ChatResponse(resposta=resultado['messages'])