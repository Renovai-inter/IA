"""Testes unitários do JudgeAgent e do LayaDecisionEngine (Laya Studio API via LAYA_API_KEY)."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.judge_agent import (
    DEFAULT_LAYA_API_URL,
    DEFAULT_LAYA_MODEL,
    FRASE_SEGURA,
    LIMIAR_NOUL_PADRAO,
    DecisionEngine,
    JudgeAgent,
    LayaDecisionEngine,
    LayaMissingKeyError,
)
from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista
from app.graph.state import EspecialistaOutput, GraphState
from app.repository.postgresql.perfil_repository import PerfilContext
from tests.support import StubAgent, StubRepository


class StubDecisionEngine(DecisionEngine):
    """Dublê de testes para o DecisionEngine, simulando o retorno da API Laya Studio."""

    def __init__(self, choice: str = "apoiada", noul: float = 0.92):
        self.choice = choice
        self.noul = noul
        self.chamadas: list[tuple[dict, dict]] = []

    def predict(self, state: dict, questions: list[dict] | dict) -> dict:
        q_norm = LayaDecisionEngine._normalizar_questions(questions)
        self.chamadas.append((state, q_norm))
        return {
            "model": DEFAULT_LAYA_MODEL,
            "answers": {
                "veredito": {
                    "type": "choice",
                    "choice": self.choice,
                    "probabilities": {
                        "apoiada": self.noul if self.choice == "apoiada" else 0.05,
                        "parcial": 0.10,
                        "sem_apoio": 0.85 if self.choice == "sem_apoio" else 0.05,
                    },
                    "confidence": 0.88,
                },
                "fidelidade": {
                    "type": "noul",
                    "noul": self.noul,
                },
            },
        }


def test_laya_decision_engine_monta_payload_e_headers_corretos(monkeypatch: pytest.MonkeyPatch):
    capturado: dict = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "multilingual",
                "answers": {
                    "veredito": {"type": "choice", "choice": "apoiada", "confidence": 0.91},
                    "fidelidade": {"type": "noul", "noul": 0.93},
                },
            }

    class _FakeClient:
        def __init__(self, timeout=None):
            capturado["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            capturado["url"] = url
            capturado["headers"] = headers
            capturado["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    engine = LayaDecisionEngine(api_key="lsk_live_teste_123")
    state_in = {"pergunta": "quanto de papelão?", "resposta": "3.450 kg", "evidencia": "3450"}
    questions_in = {
        "veredito": {
            "type": "choice",
            "instructions": "Sustentada?",
            "criteria": {"apoiada": "sim", "parcial": "parcial", "sem_apoio": "não"},
        },
        "fidelidade": {"type": "noul", "instructions": "Fiel?"},
    }

    res = engine.predict(state_in, questions_in)

    assert capturado["url"] == DEFAULT_LAYA_API_URL
    assert capturado["headers"]["Authorization"] == "Bearer lsk_live_teste_123"
    assert capturado["json"]["model"] == DEFAULT_LAYA_MODEL
    assert capturado["json"]["state"] == state_in
    assert "veredito" in capturado["json"]["questions"]
    assert "fidelidade" in capturado["json"]["questions"]
    assert res["answers"]["veredito"]["choice"] == "apoiada"
    assert res["answers"]["fidelidade"]["noul"] == 0.93


def test_laya_decision_engine_sem_chave_dispara_erro_e_judge_aplica_failsafe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    engine = LayaDecisionEngine(api_key="")
    with pytest.raises(LayaMissingKeyError):
        engine.predict({"pergunta": "oi"}, {"veredito": {"type": "noul", "instructions": "ok"}})

    agent = JudgeAgent(engine=engine)
    state: GraphState = {
        "messages": [
            HumanMessage(content="qual o saldo?"),
            AIMessage(content="O saldo é R$ 10.000,00."),
        ],
    }
    resultado = agent.run(state, config={})
    assert resultado["passou_juiz"] is False
    assert resultado["veredito_juiz"] == "sem_chave"
    assert resultado["agentes_chamados"] == ["judge:sem_chave"]
    assert resultado["messages"][0]["content"] == FRASE_SEGURA


def test_judge_agent_trata_erro_http_da_api_laya(monkeypatch: pytest.MonkeyPatch):
    class _FailingEngine(DecisionEngine):
        def predict(self, state: dict, questions: list[dict] | dict) -> dict:
            req = httpx.Request("POST", DEFAULT_LAYA_API_URL)
            resp = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("Rate limit exceeded", request=req, response=resp)

    agent = JudgeAgent(engine=_FailingEngine())
    state: GraphState = {
        "messages": [
            HumanMessage(content="qual o rateio?"),
            AIMessage(content="O rateio foi de R$ 26.400,00."),
        ],
    }
    resultado = agent.run(state, config={})
    assert resultado["passou_juiz"] is False
    assert resultado["veredito_juiz"] == "erro_http"
    assert resultado["agentes_chamados"] == ["judge:erro_http_429"]
    assert resultado["messages"][0]["content"] == FRASE_SEGURA


def test_judge_agent_aprova_resposta_fiel():
    stub_engine = StubDecisionEngine(choice="apoiada", noul=0.92)
    agent = JudgeAgent(engine=stub_engine)

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
    assert "messages" not in resultado  # aprovada -> mantém mensagem anterior intacta
    assert resultado["agentes_chamados"] == ["judge:apoiada noul=0.92"]
    assert len(stub_engine.chamadas) == 1
    enviado_state, _ = stub_engine.chamadas[0]
    assert "3450" in enviado_state["evidencia"]


def test_judge_agent_rejeita_alucinacao_e_aplica_frase_segura():
    stub_engine = StubDecisionEngine(choice="sem_apoio", noul=0.12)
    agent = JudgeAgent(engine=stub_engine)

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
    assert resultado["veredito_juiz"] == "sem_apoio"
    assert resultado["score_juiz"] < LIMIAR_NOUL_PADRAO
    assert resultado["frase_original"] == alucinacao
    assert len(resultado["messages"]) == 1
    assert resultado["messages"][0]["content"] == FRASE_SEGURA
    assert resultado["agentes_chamados"] == ["judge:sem_apoio noul=0.12"]


def test_judge_agent_rejeita_apoiada_abaixo_do_limiar():
    stub_engine = StubDecisionEngine(choice="apoiada", noul=0.55)
    agent = JudgeAgent(engine=stub_engine)

    state: GraphState = {
        "messages": [
            HumanMessage(content="qual o estoque?"),
            AIMessage(content="Temos cerca de 500 kg."),
        ],
    }
    resultado = agent.run(state, config={})

    assert resultado["passou_juiz"] is False
    assert resultado["veredito_juiz"] == "apoiada"
    assert resultado["agentes_chamados"] == ["judge:apoiada:abaixo_do_limiar noul=0.55"]
    assert resultado["messages"][0]["content"] == FRASE_SEGURA


def test_judge_agent_arun_equivalente_ao_run():
    agent = JudgeAgent(engine=StubDecisionEngine(choice="apoiada", noul=0.95))

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
        judge_agent=JudgeAgent(engine=StubDecisionEngine(choice="sem_apoio", noul=0.08)),
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

    assert "material_estoque" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert "judge:sem_apoio noul=0.08" in resultado["agentes_chamados"]
    assert resultado["passou_juiz"] is False
    assert resultado["messages"][-1].content == FRASE_SEGURA
    assert resultado["frase_original"] == "Temos 99.999 kg de papelão e 88.888 kg de vidro."


def test_gemini_decision_engine_parseia_resposta_valida():
    from app.agents.judge_agent import GeminiDecisionEngine

    texto_json = '{"choice": "apoiada", "noul": 0.95}'
    usage = {"input_tokens": 120, "output_tokens": 15}
    res = GeminiDecisionEngine._parsear_resposta(texto_json, usage_metadata=usage, model_name="gemini-3.5-flash-lite")

    assert res["model"] == "gemini-3.5-flash-lite"
    assert res["_origem"] == "gemini"
    assert res["answers"]["veredito"]["choice"] == "apoiada"
    assert res["answers"]["fidelidade"]["noul"] == 0.95
    assert res["usage"]["input_tokens"] == 120
    assert res["usage"]["output_tokens"] == 15


def test_gemini_decision_engine_sem_chave_dispara_failsafe(monkeypatch: pytest.MonkeyPatch):
    from app.agents.judge_agent import GeminiMissingKeyError

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent = JudgeAgent(gemini_api_key="")

    state: GraphState = {
        "messages": [
            HumanMessage(content="quanto papelão temos?"),
            AIMessage(content="Temos 3.450 kg."),
        ],
    }
    resultado = agent.run(state, config={})
    assert resultado["passou_juiz"] is False
    assert resultado["veredito_juiz"] == "sem_chave"
    assert resultado["agentes_chamados"] == ["judge:sem_chave"]
    assert resultado["messages"][0]["content"] == FRASE_SEGURA


def test_gemini_decision_engine_mockado_aprova_e_rejeita(monkeypatch: pytest.MonkeyPatch):
    from app.agents.judge_agent import GeminiDecisionEngine

    class _MockChat:
        def __init__(self, content):
            self.content = content
            self.usage_metadata = {"input_tokens": 80, "output_tokens": 12}

        def invoke(self, prompt):
            return self

    # Caso 1: Apoiada
    engine_aprovada = GeminiDecisionEngine(api_key="teste_key")
    monkeypatch.setattr(engine_aprovada, "_obter_llm", lambda model_name=None: _MockChat('{"choice": "apoiada", "noul": 0.95}'))
    agent_aprovado = JudgeAgent(engine=engine_aprovada)

    res_ok = agent_aprovado.run({
        "messages": [
            HumanMessage(content="qual o estoque?"),
            AIMessage(content="O estoque é 500 kg."),
        ],
    })
    assert res_ok["passou_juiz"] is True
    assert res_ok["veredito_juiz"] == "apoiada"
    assert "messages" not in res_ok

    # Caso 2: Alucinação
    engine_rejeitada = GeminiDecisionEngine(api_key="teste_key")
    monkeypatch.setattr(engine_rejeitada, "_obter_llm", lambda model_name=None: _MockChat('{"choice": "sem_apoio", "noul": 0.10}'))
    agent_rejeitado = JudgeAgent(engine=engine_rejeitada)

    res_aluc = agent_rejeitado.run({
        "messages": [
            HumanMessage(content="qual o estoque?"),
            AIMessage(content="O estoque é 999999 kg."),
        ],
    })
    assert res_aluc["passou_juiz"] is False
    assert res_aluc["veredito_juiz"] == "sem_apoio"
    assert res_aluc["messages"][0]["content"] == FRASE_SEGURA


def test_gemini_decision_engine_fallback_em_erro_transiente(monkeypatch: pytest.MonkeyPatch):
    from app.agents.judge_agent import GeminiDecisionEngine

    chamadas_modelos = []

    class _MockLLM:
        def __init__(self, model_name):
            self.model_name = model_name

        def invoke(self, prompt):
            chamadas_modelos.append(self.model_name)
            if self.model_name == "gemini-3.5-flash-lite":
                raise RuntimeError("504 DEADLINE_EXCEEDED: Deadline expired")
            # Contingência funciona
            return SimpleNamespace(
                content='{"choice": "apoiada", "noul": 0.92}',
                usage_metadata={"input_tokens": 50, "output_tokens": 10},
            )

    engine = GeminiDecisionEngine(api_key="teste_key", model="gemini-3.5-flash-lite")
    monkeypatch.setattr(engine, "_obter_llm", lambda model_name=None: _MockLLM(model_name or engine.model))

    res = engine.predict({"pergunta": "teste", "resposta": "ok", "evidencia": "ok"}, {})
    assert "gemini-3.5-flash-lite" in chamadas_modelos
    assert "gemini-2.5-flash" in chamadas_modelos
    assert res["answers"]["veredito"]["choice"] == "apoiada"
    assert res["model"] == "gemini-2.5-flash"

