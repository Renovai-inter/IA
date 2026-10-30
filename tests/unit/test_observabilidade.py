"""
Testes unitários do módulo de Observabilidade, Telemetria, SLIs/SLOs e FinOps (ROI) do Renovai.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.core.observabilidade import Observabilidade


def test_abrir_fechar_turno():
    obs = Observabilidade()
    assert obs.situacao()["em_andamento"] == 0

    t_id = obs.abrir_turno(session_id="sess-1")
    assert obs.situacao()["em_andamento"] == 1
    assert obs.turno_atual() == t_id
    assert obs.sessao_atual() == "sess-1"

    obs.fechar_turno(t_id, session_id="sess-1", rota="estoque", status="ok")
    assert obs.situacao()["em_andamento"] == 0
    assert obs.turno_atual() is None

    turnos = obs.listar_turnos()
    assert len(turnos) == 1
    assert turnos[0]["rota"] == "estoque"
    assert turnos[0]["status"] == "ok"
    assert turnos[0]["session_id"] == "sess-1"
    assert "latencia_ms" in turnos[0]


@pytest.mark.asyncio
async def test_contextvars_isolamento_assincrono():
    obs = Observabilidade()

    async def tarefa(sess_id: str):
        t_id = obs.abrir_turno(session_id=sess_id)
        assert obs.turno_atual() == t_id
        assert obs.sessao_atual() == sess_id
        await asyncio.sleep(0.01)
        assert obs.turno_atual() == t_id
        assert obs.sessao_atual() == sess_id
        obs.fechar_turno(t_id, session_id=sess_id, rota="rag_faq", status="ok")

    await asyncio.gather(
        tarefa("sess-A"),
        tarefa("sess-B"),
        tarefa("sess-C"),
    )

    assert obs.situacao()["em_andamento"] == 0
    turnos = obs.listar_turnos()
    assert len(turnos) == 3


def test_calculo_custo_usd():
    obs = Observabilidade()

    # Gemini 2.5 flash: $0.30/M in, $2.50/M out
    custo = obs.custo_usd("gemini-2.5-flash", 1_000, 1_000)
    assert pytest.approx(custo, rel=1e-5) == 0.0028

    # Gemini 3.5 flash-lite: $0.15/M in, $0.60/M out
    custo_lite = obs.custo_usd("gemini-3.5-flash-lite", 10_000, 100)
    assert pytest.approx(custo_lite, rel=1e-5) == 0.00156

    # Modelo local Ollama => $0.00
    custo_local = obs.custo_usd("ollama/llama3", 5_000, 5_000)
    assert custo_local == 0.0


def test_slis_e_slos_e_bloqueios():
    obs = Observabilidade()

    # 10 turnos: 8 ok, 1 bloqueado (guardrail), 1 erro
    for i in range(8):
        tid = obs.abrir_turno("s")
        obs.fechar_turno(tid, "s", rota="estoque", status="ok")

    tid_b = obs.abrir_turno("s")
    obs.fechar_turno(tid_b, "s", rota="bloqueio_guardrail", status="bloqueado")

    tid_e = obs.abrir_turno("s")
    obs.fechar_turno(tid_e, "s", rota="fim", status="erro")

    painel = obs.montar_painel()
    resumo = painel["resumo"]

    assert resumo["turnos"] == 10
    assert resumo["turnos_resolvidos"] == 8
    assert resumo["bloqueios"] == 1
    assert resumo["taxa_erro"] == 0.10
    assert resumo["slo_ok"] is False


def test_metricas_sre_roi_cenarios_100_e_1000_usuarios_e_latencia_interagentes():
    obs = Observabilidade()

    tid1 = obs.abrir_turno("sess-roi-1")
    obs.registrar_span("router", "grafo", 120)
    obs.registrar_span("material_estoque_agent", "grafo", 450)
    obs.registrar_span("orchestrator", "grafo", 210)
    obs.registrar_span("judge", "grafo", 85)
    obs.fechar_turno(tid1, "sess-roi-1", rota="estoque", status="ok")

    tid2 = obs.abrir_turno("sess-roi-2")
    obs.registrar_span("router", "grafo", 110)
    obs.registrar_span("financeiro_agent", "grafo", 500)
    obs.registrar_span("orchestrator", "grafo", 200)
    obs.registrar_span("judge", "grafo", 90)
    obs.fechar_turno(tid2, "sess-roi-2", rota="bloqueio_juiz", status="recusado_juiz")

    painel = obs.montar_painel()
    sre_roi = painel["sre_roi"]
    cenarios = sre_roi["cenarios_usuarios_semanais"]

    assert painel["resumo"]["recusas_juiz"] == 1
    assert sre_roi["indice_erros"]["recusas_juiz_alucinacao"] == 1
    assert sre_roi["indice_erros"]["resolvidos_com_sucesso"] == 1
    assert sre_roi["custo_por_resolucao_usd"] > 0
    assert sre_roi["roi"]["roi_percentual"] > 0
    assert sre_roi["roi"]["multiplicador_retorno_custo"] > 1

    c100 = cenarios["cenario_100_usuarios"]
    c1000 = cenarios["cenario_1000_usuarios"]
    assert c100["usuarios_semanais"] == 100
    assert c1000["usuarios_semanais"] == 1000
    assert pytest.approx(c1000["custo_semanal_usd"], rel=1e-3) == c100["custo_semanal_usd"] * 10

    agentes_medidos = {item["agente"]: item for item in painel["latencia_interagentes"]}
    assert "router" in agentes_medidos
    assert agentes_medidos["router"]["execucoes"] == 2
    assert "judge" in agentes_medidos


def test_slo_ok_quando_dentro_dos_limites():
    obs = Observabilidade()

    for i in range(20):
        tid = obs.abrir_turno("s")
        obs.fechar_turno(tid, "s", rota="estoque", status="ok")

    painel = obs.montar_painel()
    resumo = painel["resumo"]
    assert resumo["turnos"] == 20
    assert resumo["taxa_erro"] == 0.0
    assert resumo["latencia_p95_ms"] <= 8_000
    assert resumo["slo_ok"] is True


def test_ring_buffer_turnos_e_logs():
    obs = Observabilidade()

    for i in range(55):
        tid = obs.abrir_turno(f"s-{i}")
        obs.fechar_turno(tid, f"s-{i}", rota="estoque", status="ok")

    assert len(obs.listar_turnos()) == 50

    for i in range(110):
        obs.info(f"Mensagem {i}", modulo="teste")

    logs = obs.listar_logs(limite=200)
    assert len(logs) == 100
    assert logs[-1]["mensagem"] == "Mensagem 109"


def test_coletor_chamadas_callback():
    obs = Observabilidade()
    tid = obs.abrir_turno(session_id="sess-cb")

    run_id = "run-123"
    obs.coletor.on_chat_model_start(
        serialized={"kwargs": {"model_name": "gemini-2.5-flash"}},
        messages=[[]],
        run_id=run_id,
        metadata={"langgraph_node": "material_estoque_agent"},
    )

    msg = AIMessage(
        content="Olá",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    result = LLMResult(generations=[[ChatGeneration(message=msg)]])
    obs.coletor.on_llm_end(result, run_id=run_id)

    obs.fechar_turno(tid, session_id="sess-cb", rota="estoque", status="ok")

    chamadas = obs.listar_chamadas()
    assert len(chamadas) == 1
    assert chamadas[0]["modelo"] == "gemini-2.5-flash"
    assert chamadas[0]["tokens_entrada"] == 100
    assert chamadas[0]["tokens_saida"] == 50
    assert chamadas[0]["custo_usd"] > 0


def test_rota_monitor_endpoint():
    from fastapi import FastAPI
    from app.api.routes import monitor

    test_app = FastAPI()
    test_app.include_router(monitor.router)
    obs = Observabilidade()
    t_id = obs.abrir_turno("s-test")
    obs.fechar_turno(t_id, "s-test", rota="estoque", status="ok")

    test_container = MagicMock()
    test_container.observabilidade = obs
    test_app.state.container = test_container

    client = TestClient(test_app)
    resp = client.get("/monitor")
    assert resp.status_code == 200
    data = resp.json()

    assert "slo" in data
    assert "resumo" in data
    assert "sre_roi" in data
    assert "latencia_interagentes" in data
    assert "serie_latencia" in data
    assert "serie_custo" in data
    assert "por_modelo" in data
    assert "por_rota" in data
    assert "turnos" in data
    assert "logs" in data
    assert data["slo"]["latencia_p95_limite_ms"] == 8000
    assert data["slo"]["taxa_erro_limite"] == 0.05
    assert data["resumo"]["turnos"] == 1


@pytest.mark.asyncio
async def test_timed_decorator_sincrono_e_assincrono():
    from app.core.observabilidade import timed

    obs = Observabilidade()

    @timed(nome="funcao_sync", categoria="teste")
    def sincrona():
        return 42

    @timed(nome="funcao_async", categoria="teste")
    async def assincrona():
        await asyncio.sleep(0.01)
        return 99

    res_sync = sincrona()
    res_async = await assincrona()

    assert res_sync == 42
    assert res_async == 99

    logs = obs.listar_logs()
    mensagens = [l["mensagem"] for l in logs]
    assert any("'funcao_sync' executado em" in m for m in mensagens)
    assert any("'funcao_async' executado em" in m for m in mensagens)
