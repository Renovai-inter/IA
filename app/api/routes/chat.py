from app.api.schemas.chat_request import ChatRequest
from app.api.schemas.chat_response import ChatResponse
from app.core.config import Settings

from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage

from fastapi import APIRouter


settings = Settings()
router = APIRouter(tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    container = request.app.state.container

    resultado = await container.graph.ainvoke(
        {"messages": [HumanMessage(content=body.pergunta)]},
        config={"configurable": {"thread_id": body.session_id}},
    )

    return ChatResponse(resposta=resultado["messages"][-1].content)