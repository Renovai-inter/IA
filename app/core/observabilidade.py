"""
Módulo de Observabilidade Centralizado do Renovai IA.

Responsabilidades:
1. Métricas de Serviço e SRE (SLIs: latência p95, taxa de erro; SLOs: 8s, 5%).
2. Métricas de Modelos de IA (tokens entrada/saída, estimativa de custo em USD, fallbacks).
3. Logging Estruturado e Debugs unificados com propagação contextual assíncrona (via contextvars).
4. Coletor de Callbacks para modelos LangChain / LangGraph.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import math
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableLambda

_FUSO = ZoneInfo("America/Sao_Paulo")

# USD por 1 milhão de tokens: (entrada, saída).
PRECO_POR_MILHAO: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "qwen/qwen3.6-27b": (0.20, 0.60),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.10, 0.40),
    "gemini-embedding-2-preview": (0.05, 0.00),
}

_NOS_ESPECIALISTA = {"material_estoque_agent", "faq_agent", "estoque", "rag_faq"}
_PROVEDORES_ESPECIALISTA_ESPERADOS = ("gemini",)

LIMITE_TURNOS = 50
LIMITE_LOGS = 100
LIMITE_P95_MS = 8_000
LIMITE_TAXA_ERRO = 0.05

_turno_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar("turno_id", default=None)
_session_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
_instancia_observabilidade: Observabilidade | None = None


def timed(nome: str, categoria: str = "geral"):
    """
    Decorador unificado de medição de latência e telemetria para nós do grafo, tools e métodos.
    Suporta corrotinas assíncronas, funções síncronas e instâncias de RunnableLambda (LangGraph).
    Registra os logs automaticamente no logger unificado `renovai.observabilidade` e no buffer circular.
    """
    def deco(fn):
        if isinstance(fn, RunnableLambda):
            return RunnableLambda(
                deco(fn.func) if fn.func else None,
                afunc=deco(fn.afunc) if fn.afunc else None,
            )

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*a, **kw):
                t0 = time.perf_counter()
                try:
                    return await fn(*a, **kw)
                finally:
                    dur = time.perf_counter() - t0
                    if _instancia_observabilidade is not None:
                        _instancia_observabilidade.info(f"'{nome}' executado em {dur:.2f}s", modulo=categoria)
                    else:
                        t_curto = (_turno_cv.get() or "—")[:8]
                        logging.getLogger("renovai.observabilidade").info(
                            f"[{categoria}] [turno={t_curto}] '{nome}' executado em {dur:.2f}s"
                        )
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*a, **kw):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    dur = time.perf_counter() - t0
                    if _instancia_observabilidade is not None:
                        _instancia_observabilidade.info(f"'{nome}' executado em {dur:.2f}s", modulo=categoria)
                    else:
                        t_curto = (_turno_cv.get() or "—")[:8]
                        logging.getLogger("renovai.observabilidade").info(
                            f"[{categoria}] [turno={t_curto}] '{nome}' executado em {dur:.2f}s"
                        )
            return sync_wrapper

    return deco


def _agora_local() -> str:
    return datetime.now(_FUSO).strftime("%H:%M:%S")


def _nome_modelo(serialized: dict | None, invocation_params: dict | None, resposta_meta: dict | None) -> str:
    candidatos = []
    for fonte in (resposta_meta or {}, invocation_params or {}, (serialized or {}).get("kwargs") or {}):
        for chave in ("model_name", "model"):
            valor = fonte.get(chave)
            if valor:
                candidatos.append(str(valor))
    for nome in candidatos:
        if nome.startswith("models/"):
            nome = nome.split("/", 1)[1]
        return nome
    return "desconhecido"


def _no_do_grafo(metadata: dict | None) -> str:
    """Extrai o nó do grafo dos metadados do LangGraph."""
    metadata = metadata or {}
    ns = str(metadata.get("langgraph_checkpoint_ns") or "")
    candidatos: list[str] = []
    if ns:
        for parte in ns.split("|"):
            candidatos.append(parte.split(":", 1)[0])
    no = metadata.get("langgraph_node")
    if no:
        candidatos.append(str(no))
    for nome in candidatos:
        if nome in _NOS_ESPECIALISTA:
            return nome
    return candidatos[0] if candidatos else ""


def _tokens(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        usage = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }
    entrada = int(usage.get("input_tokens") or 0)
    saida = int(usage.get("output_tokens") or 0)
    if entrada or saida:
        return entrada, saida

    meta = getattr(message, "response_metadata", None) or {}
    bruto = meta.get("token_usage") or meta.get("usage") or {}
    entrada = int(bruto.get("prompt_tokens") or bruto.get("input_tokens") or 0)
    saida = int(bruto.get("completion_tokens") or bruto.get("output_tokens") or 0)
    return entrada, saida


def _eh_fallback(no: str, modelo: str) -> bool:
    if no not in _NOS_ESPECIALISTA:
        return False
    # Especialistas do Renovai (material_estoque e faq) esperam primariamente Gemini
    return not any(prov in modelo.lower() for prov in _PROVEDORES_ESPECIALISTA_ESPERADOS)


def _percentil_95(valores: list[int]) -> int:
    """Nearest-rank: o menor valor que cobre 95% das amostras, já ordenadas."""
    if not valores:
        return 0
    ordenados = sorted(valores)
    indice = math.ceil(0.95 * len(ordenados)) - 1
    return ordenados[min(max(indice, 0), len(ordenados) - 1)]


def _media(valores: list[float]) -> float:
    if not valores:
        return 0.0
    return sum(valores) / len(valores)


def _agrupar(chamadas: list[dict], chave: str) -> dict[str, list[dict]]:
    grupos: dict[str, list[dict]] = {}
    for item in chamadas:
        grupos.setdefault(item.get(chave, "—"), []).append(item)
    return grupos


class ColetorChamadas(BaseCallbackHandler):
    """CallbackHandler para capturar chamadas de LLM em tempo de execução."""

    raise_error = False

    def __init__(self, obs: Observabilidade):
        super().__init__()
        self.obs = obs

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        invocation = kwargs.get("invocation_params") or {}
        modelo = _nome_modelo(serialized, invocation, None)
        no = _no_do_grafo(metadata)
        turno_id = self.obs.turno_atual()
        session_id = self.obs.sessao_atual()

        with self.obs._lock:
            self.obs._runs[str(run_id)] = {
                "t0": time.perf_counter(),
                "modelo": modelo,
                "no": no,
                "turno_id": turno_id,
                "session_id": session_id,
            }
        self.obs.debug(f"Início LLM: modelo={modelo}, no={no or 'raiz'}", modulo="llm")

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        chave = str(run_id)
        with self.obs._lock:
            abertura = self.obs._runs.pop(chave, None)
        if abertura is None or not abertura.get("turno_id"):
            return

        message = None
        try:
            message = response.generations[0][0].message
        except (IndexError, AttributeError):
            message = None

        entrada, saida = _tokens(message) if message is not None else (0, 0)
        meta = getattr(message, "response_metadata", None) if message is not None else None
        modelo = _nome_modelo(None, None, meta)
        if modelo == "desconhecido":
            modelo = abertura["modelo"]

        latencia_ms = int((time.perf_counter() - abertura["t0"]) * 1000)
        custo = self.obs.custo_usd(modelo, entrada, saida)
        fallback = _eh_fallback(abertura["no"], modelo)

        registro = {
            "turno_id": abertura["turno_id"],
            "modelo": modelo,
            "no": abertura["no"],
            "tokens_entrada": entrada,
            "tokens_saida": saida,
            "custo_usd": custo,
            "latencia_ms": latencia_ms,
            "fallback": fallback,
        }
        with self.obs._lock:
            self.obs._chamadas.append(registro)

        self.obs.debug(
            f"Fim LLM: {modelo} ({latencia_ms}ms) | {entrada}in/{saida}out tokens, ${custo:.6f}"
            + (" [FALLBACK]" if fallback else ""),
            modulo="llm",
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        chave = str(run_id)
        with self.obs._lock:
            abertura = self.obs._runs.pop(chave, None)
        modelo = abertura.get("modelo") if abertura else "desconhecido"
        self.obs.error(f"Erro no LLM ({modelo}): {error}", modulo="llm", exc=error)


class Observabilidade:
    """Núcleo integrado de Observabilidade, Telemetria e SLIs/SLOs do Renovai."""

    def __init__(self) -> None:
        global _instancia_observabilidade
        _instancia_observabilidade = self
        self._lock = threading.Lock()
        self._inicio_processo = time.time()
        self._em_andamento = 0
        self._turnos: list[dict[str, Any]] = []
        self._chamadas: list[dict[str, Any]] = []
        self._abertos: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._logs: list[dict[str, Any]] = []
        self._modelos_acoplados: set[int] = set()

        self._logger = logging.getLogger("renovai.observabilidade")
        self._logger.setLevel(logging.DEBUG)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self.coletor = ColetorChamadas(self)

    # --------------------------------------------------------------------------
    # CONTEXTO ASSÍNCRONO
    # --------------------------------------------------------------------------
    def turno_atual(self) -> str | None:
        return _turno_cv.get()

    def sessao_atual(self) -> str | None:
        return _session_cv.get()

    # --------------------------------------------------------------------------
    # LOGGING E DEBUGS ESTRUTURADOS
    # --------------------------------------------------------------------------
    def _registrar_log(self, nivel: str, msg: str, modulo: str, exc: BaseException | None = None) -> None:
        hora = _agora_local()
        turno_id = self.turno_atual() or "—"
        session_id = self.sessao_atual() or "—"
        log_entry = {
            "hora": hora,
            "nivel": nivel,
            "modulo": modulo,
            "turno_id": turno_id,
            "session_id": session_id,
            "mensagem": str(msg),
        }
        if exc is not None:
            log_entry["erro"] = str(exc)

        with self._lock:
            self._logs.append(log_entry)
            if len(self._logs) > LIMITE_LOGS:
                self._logs.pop(0)

        t_curto = turno_id[:8] if turno_id != "—" else "—"
        texto_formatado = f"[{modulo}] [turno={t_curto}] {msg}"
        if nivel == "DEBUG":
            self._logger.debug(texto_formatado)
        elif nivel == "INFO":
            self._logger.info(texto_formatado)
        elif nivel == "WARNING":
            self._logger.warning(texto_formatado)
        elif nivel == "ERROR":
            self._logger.error(texto_formatado, exc_info=exc is not None)

    def debug(self, msg: str, modulo: str = "renovai") -> None:
        self._registrar_log("DEBUG", msg, modulo)

    def info(self, msg: str, modulo: str = "renovai") -> None:
        self._registrar_log("INFO", msg, modulo)

    def warning(self, msg: str, modulo: str = "renovai") -> None:
        self._registrar_log("WARNING", msg, modulo)

    def error(self, msg: str, modulo: str = "renovai", exc: BaseException | None = None) -> None:
        self._registrar_log("ERROR", msg, modulo, exc=exc)

    def listar_logs(self, limite: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._logs[-limite:])

    # --------------------------------------------------------------------------
    # PRECIFICAÇÃO E CUSTOS EM USD
    # --------------------------------------------------------------------------
    def custo_usd(self, modelo: str, entrada: int, saida: int) -> float:
        precos = PRECO_POR_MILHAO.get(modelo)
        if precos is None:
            for chave, valor in PRECO_POR_MILHAO.items():
                if chave in modelo:
                    precos = valor
                    break
        if precos is None:
            # Modelos locais (Ollama, qwen local) não têm custo de API
            if "ollama" in modelo.lower() or "local" in modelo.lower():
                return 0.0
            if "gemini" in modelo.lower():
                precos = PRECO_POR_MILHAO["gemini-2.5-flash"]
            elif "qwen" in modelo.lower():
                precos = PRECO_POR_MILHAO["qwen/qwen3.6-27b"]
            elif "gpt" in modelo.lower() or "groq" in modelo.lower():
                precos = PRECO_POR_MILHAO["openai/gpt-oss-120b"]
            else:
                return 0.0
        preco_entrada, preco_saida = precos
        return (entrada * preco_entrada + saida * preco_saida) / 1_000_000

    # --------------------------------------------------------------------------
    # ACOPLAMENTO DE MODELOS
    # --------------------------------------------------------------------------
    def acoplar_modelo(self, modelo: Any) -> None:
        if modelo is None:
            return
        m_id = id(modelo)
        if m_id in self._modelos_acoplados:
            return
        try:
            atuais = list(getattr(modelo, "callbacks", None) or [])
            if not any(isinstance(item, ColetorChamadas) for item in atuais):
                modelo.callbacks = [*atuais, self.coletor]
            self._modelos_acoplados.add(m_id)
        except Exception as e:
            self.debug(f"Não foi possível acoplar callback ao modelo {modelo}: {e}", modulo="observabilidade")

    def acoplar_provedores(self, providers: dict[str, Any]) -> None:
        """Percorre os provedores e tiers acoplando o callback coletor."""
        for prov_nome, prov in (providers or {}).items():
            llms_dict = getattr(prov, "_LLMs", {})
            for tier, llm_inst in llms_dict.items():
                self.acoplar_modelo(llm_inst)
        self.info("Callbacks de observabilidade vinculados aos provedores LLM.", modulo="observabilidade")

    # --------------------------------------------------------------------------
    # CICLO DE VIDA DO TURNO
    # --------------------------------------------------------------------------
    def abrir_turno(self, session_id: str | None = None) -> str:
        turno_id = uuid.uuid4().hex
        _turno_cv.set(turno_id)
        _session_cv.set(session_id)

        with self._lock:
            self._em_andamento += 1
            self._abertos[turno_id] = {
                "t0": time.perf_counter(),
                "session_id": session_id or "—",
            }

        self.info(f"Turno aberto ({turno_id[:8]}) | sessão={session_id or '—'}", modulo="turno")
        return turno_id

    def fechar_turno(self, turno_id: str, session_id: str, rota: str, status: str) -> None:
        _turno_cv.set(None)
        _session_cv.set(None)

        with self._lock:
            abertura = self._abertos.pop(turno_id, None)
            if abertura is None:
                return
            self._em_andamento = max(0, self._em_andamento - 1)
            latencia_ms = int((time.perf_counter() - abertura["t0"]) * 1000)

            do_turno = [c for c in self._chamadas if c["turno_id"] == turno_id]
            tokens_total = sum(c["tokens_entrada"] + c["tokens_saida"] for c in do_turno)
            custo_usd_total = sum(c["custo_usd"] for c in do_turno)

            registro_turno = {
                "id": turno_id,
                "hora": _agora_local(),
                "session_id": session_id or abertura.get("session_id", "—"),
                "latencia_ms": latencia_ms,
                "rota": rota or "—",
                "status": status,
                "tokens_total": tokens_total,
                "custo_usd": custo_usd_total,
            }
            self._turnos.append(registro_turno)

            while len(self._turnos) > LIMITE_TURNOS:
                antigo = self._turnos.pop(0)
                self._chamadas[:] = [c for c in self._chamadas if c["turno_id"] != antigo["id"]]

        self.info(
            f"Turno fechado ({turno_id[:8]}) | rota={rota or '—'}, status={status}, "
            f"latencia={latencia_ms}ms, tokens={tokens_total}, custo=${custo_usd_total:.6f}",
            modulo="turno",
        )

    # --------------------------------------------------------------------------
    # CONSULTAS E MÉTRICAS
    # --------------------------------------------------------------------------
    def listar_turnos(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{chave: valor for chave, valor in turno.items() if chave != "id"} for turno in self._turnos]

    def listar_chamadas(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{chave: valor for chave, valor in chamada.items() if chave != "turno_id"} for chamada in self._chamadas]

    def situacao(self) -> dict[str, int]:
        with self._lock:
            return {
                "em_andamento": self._em_andamento,
                "no_ar_segundos": int(time.time() - self._inicio_processo),
            }

    # --------------------------------------------------------------------------
    # PAINEL CONSOLIDADO (SLIs, SLOs e SÉRIES TEMPORAIS)
    # --------------------------------------------------------------------------
    def montar_painel(self) -> dict[str, Any]:
        turnos = self.listar_turnos()
        chamadas = self.listar_chamadas()
        estado = self.situacao()

        latencias = [t["latencia_ms"] for t in turnos]
        erros = sum(1 for t in turnos if t["status"] == "erro")
        bloqueios = sum(1 for t in turnos if t["status"] == "bloqueado")
        taxa_erro = (erros / len(turnos)) if turnos else 0.0
        p95 = _percentil_95(latencias)

        tokens_entrada = sum(c["tokens_entrada"] for c in chamadas)
        tokens_saida = sum(c["tokens_saida"] for c in chamadas)
        custo = sum(c["custo_usd"] for c in chamadas)

        por_modelo_bruto = _agrupar(chamadas, "modelo")
        mais_usado = "—"
        if por_modelo_bruto:
            mais_usado = max(
                por_modelo_bruto,
                key=lambda nome: (
                    len(por_modelo_bruto[nome]),
                    sum(i["tokens_entrada"] + i["tokens_saida"] for i in por_modelo_bruto[nome]),
                ),
            )

        por_modelo = []
        for modelo, itens in por_modelo_bruto.items():
            por_modelo.append({
                "modelo": modelo,
                "chamadas": len(itens),
                "tokens_entrada": sum(i["tokens_entrada"] for i in itens),
                "tokens_saida": sum(i["tokens_saida"] for i in itens),
                "custo_usd": round(sum(i["custo_usd"] for i in itens), 6),
                "latencia_media_ms": round(_media([i["latencia_ms"] for i in itens])),
            })
        por_modelo.sort(key=lambda linha: linha["custo_usd"], reverse=True)

        por_rota_bruto = _agrupar(turnos, "rota")
        por_rota = []
        for rota, itens in por_rota_bruto.items():
            por_rota.append({
                "rota": rota,
                "turnos": len(itens),
                "erros": sum(1 for i in itens if i["status"] == "erro"),
                "latencia_media_ms": round(_media([i["latencia_ms"] for i in itens])),
                "custo_usd": round(sum(i["custo_usd"] for i in itens), 6),
            })
        por_rota.sort(key=lambda linha: linha["turnos"], reverse=True)

        custo_acumulado = 0.0
        serie_custo = []
        for turno in turnos:
            custo_acumulado += turno["custo_usd"]
            serie_custo.append({
                "hora": turno["hora"],
                "custo_acumulado_usd": round(custo_acumulado, 6),
            })

        slo_ok = (p95 <= LIMITE_P95_MS) and (taxa_erro <= LIMITE_TAXA_ERRO)

        return {
            "slo": {
                "latencia_p95_limite_ms": LIMITE_P95_MS,
                "taxa_erro_limite": LIMITE_TAXA_ERRO,
            },
            "resumo": {
                "turnos": len(turnos),
                "em_andamento": estado["em_andamento"],
                "no_ar_segundos": estado["no_ar_segundos"],
                "taxa_erro": round(taxa_erro, 4),
                "bloqueios": bloqueios,
                "latencia_media_ms": round(_media(latencias)),
                "latencia_p95_ms": p95,
                "slo_ok": slo_ok,
                "tokens_entrada": tokens_entrada,
                "tokens_saida": tokens_saida,
                "tokens_total": tokens_entrada + tokens_saida,
                "custo_usd": round(custo, 6),
                "modelo_mais_usado": mais_usado,
                "fallbacks": sum(1 for c in chamadas if c["fallback"]),
            },
            "serie_latencia": [
                {"hora": t["hora"], "latencia_ms": t["latencia_ms"]} for t in turnos
            ],
            "serie_custo": serie_custo,
            "por_modelo": por_modelo,
            "por_rota": por_rota,
            "turnos": [
                {**t, "custo_usd": round(t["custo_usd"], 6)} for t in turnos
            ],
            "logs": self.listar_logs(limite=50),
        }
