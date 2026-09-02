from app.core.container import build_container
from app.api.routes.chat import router as chat_router
from app.core.config import Settings

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    container = build_container(settings)
    app.state.container = container

    yield

    container.pg_pool.close()

app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)

@app.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok"}