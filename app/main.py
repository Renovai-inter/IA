from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.container import build_container
from app.api.routes import chat, sessions, monitor
from app.core.config import Settings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

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

app = FastAPI(
    title="Renovai IA",
    description="Sistema Multi-Agente com Observabilidade Integrada",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(monitor.router)

@app.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok"}

if FRONTEND_DIR.exists():
    @app.get("/monitor.html", include_in_schema=False)
    async def monitor_page():
        return FileResponse(FRONTEND_DIR / "monitor.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")