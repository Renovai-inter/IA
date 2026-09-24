from app.core.container import build_container
from app.api.routes import chat, sessions
from app.core.config import Settings

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    container = build_container(settings)
    app.state.container = container

    yield

    for factory, conn in container.factories.values():
        try:
            factory.close(conn)
        except Exception:
            logger.exception(f"Falha ao fechar conexão de {factory.__class__.__name__}")

app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(sessions.router)

@app.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok"}