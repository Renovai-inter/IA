from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.a2a.server import router as a2a_router
from app.core.container import build_container
from app.api.routes import chat, sessions, monitor, rotas
from app.core.config import Settings
from app.core.checkpointer import abrir_checkpointer

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'
FRONTEND_DIR = _FRONTEND_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    async with abrir_checkpointer(settings) as checkpointer:
        container = build_container(settings, checkpointer=checkpointer)
        app.state.container = container
        try:
            # Pré-aquecimento do modelo leve de classificação de rotas para eliminar latência no 1º turno
            if hasattr(container, 'modelo_rotas') and hasattr(container.modelo_rotas, 'garantir_modelo'):
                try:
                    container.modelo_rotas.garantir_modelo()
                except Exception:
                    logger.warning('Não foi possível pré-aquecer o modelo de rotas no startup.')
            yield
        finally:
            if hasattr(container, 'a2a_client') and hasattr(container.a2a_client, 'aclose'):
                try:
                    await container.a2a_client.aclose()
                except Exception:
                    logger.exception('Falha ao fechar conexões HTTP do A2AClient')
            for factory, conn in container.factories.values():
                try:
                    factory.close(conn)
                except Exception:
                    logger.exception(f'Falha ao fechar conexão de {factory.__class__.__name__}')


app = FastAPI(
    title='Renovai IA',
    description='Sistema Multi-Agente com Arquitetura A2A (Cooperativas ou Empresas) e Observabilidade Integrada',
    version='0.2.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(monitor.router)
app.include_router(rotas.router)
app.include_router(a2a_router)


@app.get('/health')
async def health(request: Request) -> dict:
    return {'status': 'ok'}


if _FRONTEND_DIR.exists():
    @app.get('/monitor.html', include_in_schema=False)
    async def monitor_page():
        return FileResponse(_FRONTEND_DIR / 'monitor.html')

    app.mount('/', StaticFiles(directory=_FRONTEND_DIR, html=True), name='frontend')
