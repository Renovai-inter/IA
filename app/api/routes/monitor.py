"""
Rota de telemetria, monitoramento SRE/ROI, MCP (Model Context Protocol) e estruturas em tempo real do Redis (Rate Limiting, Fila e Ranking).

Disponibiliza:
- GET /monitor: painel completo de SRE (SLIs/SLOs), ROI, latência interagentes, LLMs, MCP e estado do Redis.
- GET /monitor/realtime: estado consolidado em tempo real do Redis (Rate Limiting, Fila e Rankings ZSET).
- GET /monitor/ranking: rankings em tempo real no Redis (entidades, rotas e usuários mais ativos).
- GET /monitor/fila: fila de processamento em tempo real no Redis (turnos ativos e últimos processados).
- GET /monitor/mcp: estatísticas de uso e inventário de ferramentas, recursos e prompts MCP.
"""

from fastapi import APIRouter, Request

from app.repository.redis.realtime_repository import RedisRealtimeRepository

router = APIRouter(tags=['monitor'])


def _obter_redis_realtime(request: Request) -> RedisRealtimeRepository | None:
    container = getattr(request.app.state, 'container', None)
    if container is None:
        return None
    repo = getattr(container, 'redis_realtime', None)
    if isinstance(repo, RedisRealtimeRepository):
        return repo
    return None


def _obter_mcp_server(request: Request):
    container = getattr(request.app.state, 'container', None)
    if container is None:
        return None
    return getattr(container, 'mcp_server', None)


@router.get('/monitor')
def painel(request: Request) -> dict:
    """Números, telemetria SRE/ROI e estruturas em tempo real do Redis no processo atual do Renovai."""
    obs = request.app.state.container.observabilidade
    dados = obs.montar_painel()
    redis_rt = _obter_redis_realtime(request)
    if redis_rt is not None:
        dados['redis_realtime'] = redis_rt.obter_painel_realtime()
    mcp_srv = _obter_mcp_server(request)
    if mcp_srv is not None and hasattr(mcp_srv, 'obter_painel_mcp'):
        dados['mcp'] = mcp_srv.obter_painel_mcp()
    return dados


@router.get('/monitor/mcp')
def painel_mcp(request: Request) -> dict:
    """Retorna estatísticas e inventário do servidor MCP (Tools, Resources, Prompts e chamadas)."""
    mcp_srv = _obter_mcp_server(request)
    if mcp_srv is not None and hasattr(mcp_srv, 'obter_painel_mcp'):
        return mcp_srv.obter_painel_mcp()
    return {'status': 'indisponivel'}


@router.get('/monitor/realtime')
def painel_realtime_redis(request: Request) -> dict:
    """Retorna o estado em tempo real do Redis: Rate Limiting (ZSET), Fila (LIST/HASH) e Rankings (ZSET)."""
    redis_rt = _obter_redis_realtime(request) or RedisRealtimeRepository()
    return redis_rt.obter_painel_realtime()


@router.get('/monitor/ranking')
def ranking_tempo_real(request: Request) -> dict:
    """Retorna os rankings em tempo real (ZSET) de Cooperativas/Empresas, Rotas e Usuários no Redis."""
    redis_rt = _obter_redis_realtime(request) or RedisRealtimeRepository()
    painel_rt = redis_rt.obter_painel_realtime()
    return {
        'backend': painel_rt['backend'],
        'rankings_tempo_real': painel_rt['rankings_tempo_real'],
    }


@router.get('/monitor/fila')
def fila_processamento_tempo_real(request: Request) -> dict:
    """Retorna a fila de processamento em tempo real (ativos e histórico recente) no Redis."""
    redis_rt = _obter_redis_realtime(request) or RedisRealtimeRepository()
    painel_rt = redis_rt.obter_painel_realtime()
    return {
        'backend': painel_rt['backend'],
        'fila_processamento': painel_rt['fila_processamento'],
    }
