"""Testes unitários do JudgeAgent e do motor de decisão de fidelidade factual."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.judge_agent import (
    FRASE_SEGURA,
    LIMIAR_NOUL_PADRAO,
    DeterministicDecisionEngine,
    JudgeAgent,
)
from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista
from app.graph.state import EspecialistaOutput, GraphState
from app.repository.postgresql.perfil_repository import PerfilContext
from tests.support import StubAgent, StubRepository


def test_deterministic_decision_engine_resposta_apoiada():
    engine = DeterministicDecisionEngine()
    pergunta = "quanto de papelão temos?"
    resposta = "Temos 3.450 kg de papelão disponível."
    evidencia = '{"material": "papelão", "quantidade_kg": 3450}'

    resultado = engine.predict(pergunta, resposta, evidencia)
    assert resultado["veredito"]["choice"] == "apoiada"
    assert resultado["fidelidade"]["noul"] >= LIMIAR_NOUL_PADRAO


def test_deterministic_decision_engine_resposta_alucinada_numeros():
    engine = DeterministicDecisionEngine()
    pergunta = "qual o valor do rateio?"
    resposta = "O rateio de setembro foi de R$ 98.750,00 e bônus de R$ 15.000,00."
    evidencia = '{"rateio_total": "26400.00"}'

    resultado = engine.predict(pergunta, resposta, evidencia)
    assert resultado["veredito"]["choice"] == "sem_apoio"
    assert resultado["fidelidade"]["noul"] < LIMIAR_NOUL_PADRAO


def test_deterministic_decision_engine_saudacoes_e_abstencoes():
    engine = DeterministicDecisionEngine()
    # Saudação
    res_saudacao = engine.predict(
        "olá",
        "Olá! Como posso ajudar com os dados da cooperativa hoje?",
        "(nenhuma)",
    )
    assert res_saudacao["veredito"]["choice"] == "apoiada"
    assert res_saudacao["fidelidade"]["noul"] >= LIMIAR_NOUL_PADRAO

    # Abstenção legítima
    res_abstencao = engine.predict(
        "quais as coletas de ontem?",
        "Não foram encontrados registros de coletas para essa data.",
        "[]",
    )
    assert res_abstencao["veredito"]["choice"] == "apoiada"
    assert res_abstencao["fidelidade"]["noul"] >= LIMIAR_NOUL_PADRAO


def test_judge_agent_aprova_resposta_fiel():
    agent = JudgeAgent(engine=DeterministicDecisionEngine())

    state: GraphState = {
        "messages": [
            HumanMessage(content="quanto papelão temos em estoque?"),
            AIMessage(content="Temos 3.450 kg de papelão disponível no estoque."),
        ],
        "snapshots": {
            "material_snapshot": SimpleNamespace(itens=[{"material": "papelão", "quantidade_kg": 3450}])
        },
        "especialista_outputs": {
            "material_estoque": EspecialistaOutput(
                conteudo='{"material": "papelão", "quantidade_kg": 3450}',
                fonte="material_estoque",
            )
        },
        "agentes_chamados": ["router", "material_estoque", "orchestrator"],
    }

    resultado = agent.run(state, config={})

    assert resultado["passou_juiz"] is True
    assert resultado["veredito_juiz"] == "apoiada"
    assert resultado["score_juiz"] >= LIMIAR_NOUL_PADRAO
    assert "messages" not in resultado  # aprovada -> mantém mensagem anterior
    assert any("judge:apoiada" in a for a in resultado["agentes_chamados"])


def test_judge_agent_rejeita_alucinacao_e_aplica_frase_segura():
    agent = JudgeAgent(engine=DeterministicDecisionEngine())

    alucinacao = "O rateio de setembro foi de R$ 98.750,00 e o bônus de R$ 15.000,00."
    state: GraphState = {
        "messages": [
            HumanMessage(content="qual o rateio de setembro?"),
            AIMessage(content=alucinacao),
        ],
        "snapshots": {
            "rateio_snapshot": SimpleNamespace(itens=[{"valor_liquido": 26400.00}])
        },
        "especialista_outputs": {
            "financeiro": EspecialistaOutput(
                conteudo='{"valor_liquido": 26400.00}',
                fonte="financeiro",
            )
        },
        "agentes_chamados": ["router", "financeiro", "orchestrator"],
    }

    resultado = agent.run(state, config={})

    assert resultado["passou_juiz"] is False
    assert resultado["veredito_juiz"] in ["sem_apoio", "parcial"]
    assert resultado["score_juiz"] < LIMIAR_NOUL_PADRAO
    assert resultado["frase_original"] == alucinacao
    assert len(resultado["messages"]) == 1
    assert resultado["messages"][0]["content"] == FRASE_SEGURA
    assert any("judge:" in a for a in resultado["agentes_chamados"])


def test_judge_agent_arun_equivalente_ao_run():
    agent = JudgeAgent(engine=DeterministicDecisionEngine())

    state: GraphState = {
        "messages": [
            HumanMessage(content="olá"),
            AIMessage(content="Olá, como posso ajudar?"),
        ],
        "snapshots": {},
        "especialista_outputs": {},
        "agentes_chamados": ["router"],
    }

    coro = agent.arun(state, config={})
    try:
        coro.send(None)
        resultado = None
    except StopIteration as e:
        resultado = e.value

    assert resultado is not None
    assert resultado["passou_juiz"] is True
    assert resultado["veredito_juiz"] == "apoiada"
    assert "messages" not in resultado


def test_grafo_executa_juiz_no_portao_terminal():
    """Valida que o nó 'judge' executa após o orquestrador/guardrail e barra alucinações ponta a ponta."""
    router = StubAgent(
        "router",
        [
            {
                "messages": [],
                "proximos_agentes": ["estoque"],
                "agentes_chamados": ["router"],
            }
        ],
    )
    # Especialista retorna evidência real (3450 kg)
    especialista = StubAgent(
        "material_estoque",
        [
            {
                "messages": [],
                "agentes_chamados": ["material_estoque"],
                "especialista_outputs": {
                    "material_estoque": EspecialistaOutput(
                        conteudo='{"quantidade_kg": 3450}',
                        fonte="material_estoque",
                    )
                },
            }
        ],
    )
    # Orquestrador alucina número não presente na evidência (99999 kg)
    orchestrator_alucinando = StubAgent(
        "orchestrator",
        [
            {
                "messages": [AIMessage(content="Temos 99.999 kg de papelão e 88.888 kg de vidro.")],
                "agentes_chamados": ["orchestrator"],
            }
        ],
    )

    builder = GraphBuilder(
        router_agent=router,
        orchestrator_agent=orchestrator_alucinando,
        specialists=[
            RegistroEspecialista(
                rota="estoque",
                node_name="material_estoque_agent",
                agente=especialista,
                repositories=["material_repository"],
            )
        ],
        repositories={"material_repository": StubRepository(snapshot={"itens": []})},
        checkpointer=MemorySaver(),
        guardrail_entrada=StubAgent("guardrail_entrada", [{"passou_guardrail": True}]),
        guardrail_saida=StubAgent("guardrail_saida", [{}]),
        judge_agent=JudgeAgent(engine=DeterministicDecisionEngine()),
    )
    graph = builder.build_graph()

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="quanto material temos?")],
            "perfil_ctx": PerfilContext(
                perfil_id=uuid4(),
                tipo="COOPERATIVA",
                empresa_id=None,
                cooperativa_id=uuid4(),
            ),
        },
        config={"configurable": {"thread_id": "teste-juiz-terminal"}},
    )

    # Verifica execução em ordem
    assert "material_estoque" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert any("judge:" in a for a in resultado["agentes_chamados"])

    # Alucinação foi barrada e substituída pela frase segura
    assert resultado["passou_juiz"] is False
    assert resultado["messages"][-1].content == FRASE_SEGURA
    assert resultado["frase_original"] == "Temos 99.999 kg de papelão e 88.888 kg de vidro."
