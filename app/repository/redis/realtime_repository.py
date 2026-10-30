"""
Repositório Redis em Tempo Real:
1. Rate Limiting por Janela Deslizante (Sliding Window Log via Redis Sorted Set `ZSET`: `ZREMRANGEBYSCORE`, `ZCARD`, `ZADD`, `EXPIRE`).
2. Fila de Processamento em Tempo Real (via Redis List `LIST`: `LPUSH`, `LTRIM`, `LRANGE` e Hash `HSET`/`HDEL`).
3. Ranking em Tempo Real (Leaderboard via Redis Sorted Set `ZSET`: `ZINCRBY`, `ZREVRANGE`).

Atende simultaneamente:
- Proteção de cota de APIs de IA e prevenção de abuso (Rate Limiting HTTP 429);
- Requisito obrigatório de Banco de Dados 2 (uso do Redis para fila de processamento e ranking em tempo real).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from redis import Redis

from app.repository.base import Repository

log = logging.getLogger("renovai.redis")
_FUSO = ZoneInfo("America/Sao_Paulo")


def _agora_iso() -> str:
    return datetime.now(_FUSO).strftime("%H:%M:%S")


@dataclass(frozen=True)
class RateLimitResult:
    permitido: bool
    identificador: str
    limite: int
    restante: int
    usado_na_janela: int
    janela_segundos: int
    retry_after_segundos: int
    backend: str


class RedisRealtimeRepository(Repository[Any, Redis | None]):
    """Gerencia Rate Limiting (ZSET), Fila de Processamento (LIST/HASH) e Ranking em Tempo Real (ZSET) no Redis."""

    KEY_PREFIX_RATELIMIT = "renovai:ratelimit"
    KEY_RATELIMIT_BLOQUEIOS = "renovai:metrics:ratelimit_bloqueios"
    KEY_FILA_ATIVOS = "renovai:fila:ativos"
    KEY_FILA_HISTORICO = "renovai:fila:processamento"
    KEY_RANKING_ENTIDADES = "renovai:ranking:entidades"
    KEY_RANKING_ROTAS = "renovai:ranking:rotas"
    KEY_RANKING_USUARIOS = "renovai:ranking:usuarios"
    MAX_HISTORICO_FILA = 50

    def __init__(
        self,
        db: Redis | None = None,
        max_requests: int = 15,
        window_seconds: int = 60,
    ):
        super().__init__(db)
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))

        # Estruturas em memória para fallback transparente quando REDIS_URL estiver vazio ou inacessível
        self._lock = threading.Lock()
        self._mem_ratelimit: dict[str, list[float]] = {}
        self._mem_bloqueios: int = 0
        self._mem_fila_ativos: dict[str, dict[str, Any]] = {}
        self._mem_fila_historico: list[dict[str, Any]] = []
        self._mem_ranking_entidades: dict[str, float] = {}
        self._mem_ranking_rotas: dict[str, float] = {}
        self._mem_ranking_usuarios: dict[str, float] = {}

    @property
    def backend_nome(self) -> str:
        return "redis" if self._db is not None else "memoria_fallback"

    # ==========================================================================
    # 1. RATE LIMITING — SLIDING WINDOW LOG (Redis ZSET)
    # ==========================================================================
    def verificar_rate_limit(
        self,
        identificador: str,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> RateLimitResult:
        """Verifica e registra uma requisição na janela deslizante do identificador informado."""
        chave_id = (identificador or "anonimo").strip() or "anonimo"
        limite = int(max_requests if max_requests is not None else self.max_requests)
        janela = int(window_seconds if window_seconds is not None else self.window_seconds)
        agora = time.time()
        inicio_janela = agora - janela

        if self._db is not None:
            try:
                return self._verificar_rate_limit_redis(chave_id, limite, janela, agora, inicio_janela)
            except Exception as exc:
                log.warning(f"Falha no Redis durante verificar_rate_limit ({exc}). Usando fallback em memória.")

        return self._verificar_rate_limit_memoria(chave_id, limite, janela, agora, inicio_janela)

    def _verificar_rate_limit_redis(
        self,
        chave_id: str,
        limite: int,
        janela: int,
        agora: float,
        inicio_janela: float,
    ) -> RateLimitResult:
        assert self._db is not None
        redis_key = f"{self.KEY_PREFIX_RATELIMIT}:{chave_id}"

        pipe = self._db.pipeline(transaction=True)
        pipe.zremrangebyscore(redis_key, "-inf", inicio_janela)
        pipe.zcard(redis_key)
        pipe.zrange(redis_key, 0, 0, withscores=True)
        _, qtd_atual, mais_antigo = pipe.execute()

        usado = int(qtd_atual or 0)
        if usado >= limite:
            self._db.incr(self.KEY_RATELIMIT_BLOQUEIOS)
            retry_after = janela
            if mais_antigo:
                ts_antigo = float(mais_antigo[0][1])
                retry_after = max(1, math.ceil((ts_antigo + janela) - agora))
            return RateLimitResult(
                permitido=False,
                identificador=chave_id,
                limite=limite,
                restante=0,
                usado_na_janela=usado,
                janela_segundos=janela,
                retry_after_segundos=retry_after,
                backend="redis",
            )

        membro = f"{agora:.6f}:{uuid.uuid4().hex[:8]}"
        pipe_add = self._db.pipeline(transaction=True)
        pipe_add.zadd(redis_key, {membro: agora})
        pipe_add.expire(redis_key, janela + 10)
        pipe_add.execute()

        usado_apos = usado + 1
        return RateLimitResult(
            permitido=True,
            identificador=chave_id,
            limite=limite,
            restante=max(0, limite - usado_apos),
            usado_na_janela=usado_apos,
            janela_segundos=janela,
            retry_after_segundos=0,
            backend="redis",
        )

    def _verificar_rate_limit_memoria(
        self,
        chave_id: str,
        limite: int,
        janela: int,
        agora: float,
        inicio_janela: float,
    ) -> RateLimitResult:
        with self._lock:
            historico = [ts for ts in self._mem_ratelimit.get(chave_id, []) if ts > inicio_janela]
            usado = len(historico)
            if usado >= limite:
                self._mem_ratelimit[chave_id] = historico
                self._mem_bloqueios += 1
                ts_antigo = historico[0] if historico else agora
                retry_after = max(1, math.ceil((ts_antigo + janela) - agora))
                return RateLimitResult(
                    permitido=False,
                    identificador=chave_id,
                    limite=limite,
                    restante=0,
                    usado_na_janela=usado,
                    janela_segundos=janela,
                    retry_after_segundos=retry_after,
                    backend="memoria_fallback",
                )

            historico.append(agora)
            self._mem_ratelimit[chave_id] = historico
            usado_apos = len(historico)
            return RateLimitResult(
                permitido=True,
                identificador=chave_id,
                limite=limite,
                restante=max(0, limite - usado_apos),
                usado_na_janela=usado_apos,
                janela_segundos=janela,
                retry_after_segundos=0,
                backend="memoria_fallback",
            )

    # ==========================================================================
    # 2. FILA DE PROCESSAMENTO EM TEMPO REAL (Redis HASH + LIST)
    # ==========================================================================
    def enfileirar_turno(
        self,
        turno_id: str,
        session_id: str,
        user_id: str | None = None,
        entidade_id: str | None = None,
        tipo_fluxo: str = "COOPERATIVA",
        pergunta: str = "",
    ) -> dict[str, Any]:
        """Insere um turno na fila de processamento ativo do Redis e registra o evento de entrada."""
        item = {
            "turno_id": turno_id[:8],
            "session_id": session_id or "—",
            "user_id": user_id or "anonimo",
            "entidade_id": entidade_id or "—",
            "tipo_fluxo": tipo_fluxo,
            "pergunta_resumo": (pergunta or "")[:80],
            "estado": "em_processamento",
            "rota": "aguardando_roteador",
            "latencia_ms": 0,
            "atualizado_em": _agora_iso(),
        }
        payload = json.dumps(item, ensure_ascii=False)

        if self._db is not None:
            try:
                pipe = self._db.pipeline(transaction=False)
                pipe.hset(self.KEY_FILA_ATIVOS, turno_id, payload)
                pipe.expire(self.KEY_FILA_ATIVOS, 3600)
                pipe.execute()
                return item
            except Exception as exc:
                log.warning(f"Falha ao enfileirar turno no Redis ({exc}). Usando fallback em memória.")

        with self._lock:
            self._mem_fila_ativos[turno_id] = item
        return item

    def finalizar_turno_fila(
        self,
        turno_id: str,
        session_id: str,
        rota: str,
        status: str,
        latencia_ms: int,
        user_id: str | None = None,
        entidade_id: str | None = None,
        tipo_fluxo: str = "COOPERATIVA",
    ) -> dict[str, Any]:
        """Remove o turno da fila de ativos e empurra (`LPUSH` + `LTRIM`) para a fila de histórico em tempo real."""
        item_concluido = {
            "turno_id": turno_id[:8],
            "session_id": session_id or "—",
            "user_id": user_id or "anonimo",
            "entidade_id": entidade_id or "—",
            "tipo_fluxo": tipo_fluxo,
            "estado": status,
            "rota": rota or "—",
            "latencia_ms": int(latencia_ms),
            "atualizado_em": _agora_iso(),
        }
        payload = json.dumps(item_concluido, ensure_ascii=False)

        if self._db is not None:
            try:
                pipe = self._db.pipeline(transaction=False)
                pipe.hdel(self.KEY_FILA_ATIVOS, turno_id)
                pipe.lpush(self.KEY_FILA_HISTORICO, payload)
                pipe.ltrim(self.KEY_FILA_HISTORICO, 0, self.MAX_HISTORICO_FILA - 1)
                pipe.expire(self.KEY_FILA_HISTORICO, 86400)
                pipe.execute()
                return item_concluido
            except Exception as exc:
                log.warning(f"Falha ao finalizar turno na fila Redis ({exc}). Usando fallback em memória.")

        with self._lock:
            self._mem_fila_ativos.pop(turno_id, None)
            self._mem_fila_historico.insert(0, item_concluido)
            if len(self._mem_fila_historico) > self.MAX_HISTORICO_FILA:
                self._mem_fila_historico = self._mem_fila_historico[: self.MAX_HISTORICO_FILA]
        return item_concluido

    # ==========================================================================
    # 3. RANKING EM TEMPO REAL (Redis ZSET: ZINCRBY + ZREVRANGE)
    # ==========================================================================
    def registrar_interacao_ranking(
        self,
        entidade_id: str | None = None,
        tipo_fluxo: str = "COOPERATIVA",
        rotas: list[str] | str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Atualiza atomicamente os rankings em tempo real no Redis via `ZINCRBY`."""
        chave_entidade = f"{tipo_fluxo}:{entidade_id}" if entidade_id else None
        chave_usuario = (user_id or "").strip() or None

        lista_rotas: list[str] = []
        if isinstance(rotas, str):
            lista_rotas = [r.strip() for r in rotas.split(",") if r.strip() and r.strip() != "—"]
        elif isinstance(rotas, list):
            lista_rotas = [str(r).strip() for r in rotas if str(r).strip()]

        if self._db is not None:
            try:
                pipe = self._db.pipeline(transaction=False)
                if chave_entidade:
                    pipe.zincrby(self.KEY_RANKING_ENTIDADES, 1.0, chave_entidade)
                if chave_usuario:
                    pipe.zincrby(self.KEY_RANKING_USUARIOS, 1.0, chave_usuario)
                for r in lista_rotas:
                    pipe.zincrby(self.KEY_RANKING_ROTAS, 1.0, r)
                pipe.execute()
                return
            except Exception as exc:
                log.warning(f"Falha ao atualizar ranking no Redis ({exc}). Usando fallback em memória.")

        with self._lock:
            if chave_entidade:
                self._mem_ranking_entidades[chave_entidade] = (
                    self._mem_ranking_entidades.get(chave_entidade, 0.0) + 1.0
                )
            if chave_usuario:
                self._mem_ranking_usuarios[chave_usuario] = (
                    self._mem_ranking_usuarios.get(chave_usuario, 0.0) + 1.0
                )
            for r in lista_rotas:
                self._mem_ranking_rotas[r] = self._mem_ranking_rotas.get(r, 0.0) + 1.0

    # ==========================================================================
    # 4. CONSULTA CONSOLIDADA (FILA, RANKINGS E RATE LIMIT)
    # ==========================================================================
    @staticmethod
    def _formatar_zset(pares: list[tuple[str, float]]) -> list[dict[str, Any]]:
        return [
            {
                "posicao": idx + 1,
                "membro": str(membro),
                "interacoes": int(score),
            }
            for idx, (membro, score) in enumerate(pares)
        ]

    def obter_painel_realtime(self, top_n: int = 10, limite_fila: int = 15) -> dict[str, Any]:
        """Retorna o estado em tempo real da fila de processamento, dos rankings e do rate limiting."""
        if self._db is not None:
            try:
                pipe = self._db.pipeline(transaction=False)
                pipe.get(self.KEY_RATELIMIT_BLOQUEIOS)
                pipe.hvals(self.KEY_FILA_ATIVOS)
                pipe.lrange(self.KEY_FILA_HISTORICO, 0, limite_fila - 1)
                pipe.zrevrange(self.KEY_RANKING_ENTIDADES, 0, top_n - 1, withscores=True)
                pipe.zrevrange(self.KEY_RANKING_ROTAS, 0, top_n - 1, withscores=True)
                pipe.zrevrange(self.KEY_RANKING_USUARIOS, 0, top_n - 1, withscores=True)
                (
                    bloqueios_raw,
                    ativos_raw,
                    historico_raw,
                    rank_entidades,
                    rank_rotas,
                    rank_usuarios,
                ) = pipe.execute()

                ativos = []
                for item_str in ativos_raw or []:
                    try:
                        ativos.append(json.loads(item_str))
                    except Exception:
                        pass

                historico = []
                for item_str in historico_raw or []:
                    try:
                        historico.append(json.loads(item_str))
                    except Exception:
                        pass

                return {
                    "backend": "redis",
                    "rate_limit": {
                        "algoritmo": "Sliding Window Log (Redis ZSET)",
                        "max_requests": self.max_requests,
                        "window_seconds": self.window_seconds,
                        "bloqueios_totais": int(bloqueios_raw or 0),
                    },
                    "fila_processamento": {
                        "tamanho_fila_ativa": len(ativos),
                        "em_processamento_agora": ativos,
                        "ultimos_processados": historico,
                    },
                    "rankings_tempo_real": {
                        "entidades_mais_ativas": self._formatar_zset(rank_entidades or []),
                        "rotas_mais_demandadas": self._formatar_zset(rank_rotas or []),
                        "usuarios_mais_ativos": self._formatar_zset(rank_usuarios or []),
                    },
                }
            except Exception as exc:
                log.warning(f"Falha ao ler painel realtime do Redis ({exc}). Usando fallback em memória.")

        with self._lock:
            ativos = list(self._mem_fila_ativos.values())
            historico = list(self._mem_fila_historico[:limite_fila])
            ent_sorted = sorted(self._mem_ranking_entidades.items(), key=lambda x: x[1], reverse=True)[:top_n]
            rot_sorted = sorted(self._mem_ranking_rotas.items(), key=lambda x: x[1], reverse=True)[:top_n]
            usr_sorted = sorted(self._mem_ranking_usuarios.items(), key=lambda x: x[1], reverse=True)[:top_n]

            return {
                "backend": "memoria_fallback",
                "rate_limit": {
                    "algoritmo": "Sliding Window Log (Fallback Memória)",
                    "max_requests": self.max_requests,
                    "window_seconds": self.window_seconds,
                    "bloqueios_totais": self._mem_bloqueios,
                },
                "fila_processamento": {
                    "tamanho_fila_ativa": len(ativos),
                    "em_processamento_agora": ativos,
                    "ultimos_processados": historico,
                },
                "rankings_tempo_real": {
                    "entidades_mais_ativas": self._formatar_zset(ent_sorted),
                    "rotas_mais_demandadas": self._formatar_zset(rot_sorted),
                    "usuarios_mais_ativos": self._formatar_zset(usr_sorted),
                },
            }
