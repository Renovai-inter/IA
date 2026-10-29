"""Testes unitários para a Arquitetura A2A com Grafo Principal Unificado (Host Graph):
1. Contrato `ChatRequest` (`cooperativa_id` / `empresa_id`) e `resolver_contexto_dominio`;
2. Agent Cards de `/a2a/cooperativa/.well-known/agent.json` (`estoque`, `coleta`, `financeiro`, `logistica`)
   e `/a2a/empresa/.well-known/agent.json` (`pedidos`);
3. Execução JSON-RPC 2.0 (`tasks/send`, `tasks/get`, `tasks/cancel`) em `A2ACooperativaServer`
   (`POST /a2a/cooperativa`) e `A2AEmpresaServer` (`POST /a2a/empresa`);
4. Grafo Principal Unificado (Host Graph):
   - `guardrail_entrada` -> `router_agent` -> [`rag_faq` local | `/a2a/cooperativa` | `/a2a/empresa`] -> `orchestrator_agent` -> `guardrail_saida` -> `judge`
   - Uso direto de `cooperativa_id` e `empresa_id` nas queries dos repositórios/tools.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from app.a2a.agent_card import get_cooperativa_agent_card, get_empresa_agent_card
from app.a2a.client import A2AClient
from app.a2a.schemas import JSONRPCRequest
from app.a2a.server import A2ACooperativaServer, A2AEmpresaServer
from app.api.schemas.chat_request import ChatRequest
from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista
from app.graph.state import EspecialistaOutput, resolver_contexto_dominio
from app.repository.postgresql.perfil_repository import PerfilContext
from app.tools.recomendacao_tools import RecomendacaoToolkit
from tests.support import StubAgent, StubRepository


# ==============================================================================
# 1. Contrato ChatRequest (cooperativa_id vs empresa_id) e Contexto Direto
# ==============================================================================
def test_chat_request_aceita_cooperativa_id():
    coop_id = str(uuid4())
    req = ChatRequest(
        session_id="sess-1",
        cooperativa_id=coop_id,
        pergunta="Quanto de PET temos no estoque?",
    )
    assert req.tipo_fluxo == "COOPERATIVA"
    assert req.entidade_id == coop_id
    assert req.empresa_id is None


def test_chat_request_aceita_empresa_id():
    emp_id = str(uuid4())
    req = ChatRequest(
        session_id="sess-2",
        empresa_id=emp_id,
        pergunta="Quais cooperativas têm 1000 kg de papelão?",
    )
    assert req.tipo_fluxo == "EMPRESA"
    assert req.entidade_id == emp_id
    assert req.cooperativa_id is None


def test_chat_request_rejeita_ambos_ou_nenhum():
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id="sess-3",
            pergunta="Sem cooperativa_id nem empresa_id",
        )

    with pytest.raises(ValidationError):
        ChatRequest(
            session_id="sess-4",
            cooperativa_id=str(uuid4()),
            empresa_id=str(uuid4()),
            pergunta="Com ambos preenchidos",
        )


def test_resolver_contexto_dominio_direto_do_state_ou_config():
    coop_uuid = uuid4()
    c_id, e_id, tipo = resolver_contexto_dominio({"cooperativa_id": coop_uuid})
    assert c_id == coop_uuid
    assert e_id is None
    assert tipo == "COOPERATIVA"

    emp_uuid = uuid4()
    c_id2, e_id2, tipo2 = resolver_contexto_dominio(
        {}, {"configurable": {"empresa_id": str(emp_uuid)}}
    )
    assert c_id2 is None
    assert e_id2 == emp_uuid
    assert tipo2 == "EMPRESA"


# ==============================================================================
# 2. Agent Cards e Servidores A2A (/a2a/cooperativa e /a2a/empresa)
# ==============================================================================
def test_agent_cards_expondo_subgrafos_cooperativa_e_empresa():
    card_coop = get_cooperativa_agent_card("http://cooperativa:8000/a2a/cooperativa")
    assert card_coop.protocolVersion == "0.2.1"
    assert {s.id for s in card_coop.skills} == {"estoque", "coleta", "financeiro", "logistica"}

    card_emp = get_empresa_agent_card("http://empresa:8000/a2a/empresa")
    assert card_emp.protocolVersion == "0.2.1"
    assert {s.id for s in card_emp.skills} == {"pedidos"}


def test_a2a_servers_executam_skills_cooperativa_e_empresa_via_jsonrpc():
    coop_uuid = uuid4()
    emp_uuid = uuid4()

    class _TrackingRepo:
        def __init__(self):
            self.cooperativa_ids_recebidos = []

        def get_snapshot(self, cooperativa_id):
            self.cooperativa_ids_recebidos.append(cooperativa_id)
            return {"itens": ["lote-pet"]}

    repo_estoque = _TrackingRepo()

    agente_estoque = StubAgent(
        "material_estoque",
        [
            {
                "messages": [{"role": "assistant", "content": '{"dominio":"material_estoque","resposta":"500kg PET"}'}],
                "agentes_chamados": ["material_estoque"],
                "especialista_outputs": {
                    "material_estoque": EspecialistaOutput(
                        conteudo='{"dominio":"material_estoque","resposta":"500kg PET"}',
                        fonte="material_estoque",
                    )
                },
            }
        ],
    )
    agente_pedidos = StubAgent(
        "recomendacao",
        [
            {
                "messages": [{"role": "assistant", "content": '{"dominio":"recomendacao","resposta":"Coop Verde 12km"}'}],
                "agentes_chamados": ["recomendacao"],
                "especialista_outputs": {
                    "recomendacao": EspecialistaOutput(
                        conteudo='{"dominio":"recomendacao","resposta":"Coop Verde 12km"}',
                        fonte="recomendacao",
                    )
                },
            }
        ],
    )

    coop_server = A2ACooperativaServer(
        specialists=[
            RegistroEspecialista(
                rota="estoque",
                node_name="material_estoque_agent",
                agente=agente_estoque,
                repositories=["estoque_repository"],
            )
        ],
        repositories={"estoque_repository": repo_estoque},
    )
    emp_server = A2AEmpresaServer(
        specialists=[
            RegistroEspecialista(
                rota="pedidos",
                node_name="recomendacao_agent",
                agente=agente_pedidos,
                repositories=[],
            )
        ],
        repositories={},
    )

    client = A2AClient(
        cooperativa_server=coop_server,
        empresa_server=emp_server,
    )

    # 1. Roteamento automático de 'estoque' para o serviço A2A Cooperativa com cooperativa_id
    resp_coop = client.send_task(
        skill="estoque",
        cooperativa_id=str(coop_uuid),
        messages=[HumanMessage(content="Quanto PET temos?")],
        session_id="sess-coop-1",
    )
    dados_coop = A2AClient.extrair_dados_artifact(resp_coop)
    assert dados_coop["status"] == "ok"
    assert dados_coop["dominio_a2a"] == "COOPERATIVA"
    assert dados_coop["nome_agente"] == "material_estoque"
    assert repo_estoque.cooperativa_ids_recebidos == [coop_uuid]

    # Consulta tasks/get no servidor da Cooperativa
    get_coop = coop_server.handle_jsonrpc(
        JSONRPCRequest(id="g1", method="tasks/get", params={"id": dados_coop["task_id"]})
    )
    assert get_coop.result.status.state == "completed"

    # 2. Roteamento automático de 'pedidos' para o serviço A2A Empresa com empresa_id
    resp_emp = client.send_task(
        skill="pedidos",
        empresa_id=str(emp_uuid),
        messages=[HumanMessage(content="Recomende cooperativas de PET")],
        session_id="sess-emp-1",
    )
    dados_emp = A2AClient.extrair_dados_artifact(resp_emp)
    assert dados_emp["status"] == "ok"
    assert dados_emp["dominio_a2a"] == "EMPRESA"
    assert dados_emp["nome_agente"] == "recomendacao"


# ==============================================================================
# 3. Host Graph Unificado acionando RAG local e Microsserviços A2A
# ==============================================================================
def test_host_graph_unificado_aciona_a2a_cooperativa_e_rag_faq_em_paralelo():
    coop_uuid = uuid4()

    router = StubAgent(
        "router",
        [
            {
                "messages": [],
                "proximos_agentes": ["estoque", "rag_faq"],
                "agentes_chamados": ["router"],
            }
        ],
    )
    especialista_estoque = StubAgent(
        "material_estoque",
        [
            {
                "messages": [{"role": "assistant", "content": "800kg de PET"}],
                "agentes_chamados": ["material_estoque"],
                "especialista_outputs": {
                    "material_estoque": EspecialistaOutput(conteudo="800kg de PET", fonte="material_estoque")
                },
            }
        ],
    )
    especialista_faq = StubAgent(
        "faq",
        [
            {
                "messages": [{"role": "assistant", "content": "Art. 9 da PNRS"}],
                "agentes_chamados": ["faq"],
                "especialista_outputs": {
                    "faq": EspecialistaOutput(conteudo="Art. 9 da PNRS", fonte="faq")
                },
            }
        ],
    )
    orchestrator = StubAgent(
        "orchestrator",
        [
            {
                "messages": [{"role": "assistant", "content": "Temos 800kg de PET segundo o Art. 9 da PNRS."}],
                "agentes_chamados": ["orchestrator"],
            }
        ],
    )

    coop_server = A2ACooperativaServer(
        specialists=[
            RegistroEspecialista(
                rota="estoque",
                node_name="material_estoque_agent",
                agente=especialista_estoque,
                repositories=["estoque_repository"],
            )
        ],
        repositories={"estoque_repository": StubRepository(snapshot={"itens": []})},
    )
    emp_server = A2AEmpresaServer(specialists=[], repositories={})
    a2a_client = A2AClient(cooperativa_server=coop_server, empresa_server=emp_server)

    graph = GraphBuilder(
        router_agent=router,
        orchestrator_agent=orchestrator,
        specialists=[
            RegistroEspecialista(
                rota="estoque",
                node_name="material_estoque_agent",
                agente=especialista_estoque,
                repositories=["estoque_repository"],
            ),
            RegistroEspecialista(
                rota="rag_faq",
                node_name="faq_agent",
                agente=especialista_faq,
                repositories=[],
            ),
        ],
        repositories={"estoque_repository": StubRepository(snapshot={"itens": []})},
        checkpointer=MemorySaver(),
        guardrail_entrada=StubAgent("guardrail_entrada", [{"passou_guardrail": True}]),
        guardrail_saida=StubAgent("guardrail_saida", [{"agentes_chamados": ["guardrail_saida:ok"]}]),
        a2a_client=a2a_client,
        judge_agent=StubAgent("judge", [{"passou_juiz": True, "agentes_chamados": ["judge:apoiada noul=0.95"]}]),
    ).build_graph()

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="Quanto PET temos e o que diz a PNRS?")],
            "cooperativa_id": coop_uuid,
        },
        config={"configurable": {"thread_id": "t-host-1", "cooperativa_id": str(coop_uuid)}},
    )

    assert "router" in resultado["agentes_chamados"]
    assert "material_estoque" in resultado["agentes_chamados"]
    assert "faq" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert "guardrail_saida:ok" in resultado["agentes_chamados"]
    assert len(coop_server._tasks) == 1


def test_recomendacao_toolkit_usa_empresa_id_do_config():
    emp_uuid = uuid4()

    class _FakeRecomendacaoRepo:
        def __init__(self):
            self.ultimo_empresa_id = None

        def recomendar_cooperativas(self, material, empresa_id=None, raio_max_km=40.0, volume_minimo_kg=500.0):
            self.ultimo_empresa_id = empresa_id
            return []

    fake_neo4j = _FakeRecomendacaoRepo()
    toolkit = RecomendacaoToolkit(repository=fake_neo4j)

    toolkit.recomendar_cooperativas(
        material="PET",
        config={"configurable": {"empresa_id": str(emp_uuid)}},
    )
    assert fake_neo4j.ultimo_empresa_id == str(emp_uuid)


# ==============================================================================
# 4. Fluxo Host Graph com Roteamento Dinâmico para /a2a/cooperativa e /a2a/empresa
# ==============================================================================
def test_host_graph_roteia_cooperativa_id_e_empresa_id_para_servicos_a2a_distintos():
    coop_uuid = uuid4()
    emp_uuid = uuid4()

    agente_coleta = StubAgent(
        "coleta",
        [
            {
                "messages": [{"role": "assistant", "content": '{"dominio":"coleta","resposta":"3 coletas hoje"}'}],
                "agentes_chamados": ["coleta"],
                "especialista_outputs": {
                    "coleta": EspecialistaOutput(conteudo="3 coletas hoje", fonte="coleta")
                },
            }
        ],
    )
    agente_pedidos = StubAgent(
        "recomendacao",
        [
            {
                "messages": [{"role": "assistant", "content": '{"dominio":"recomendacao","resposta":"Coop Top 1"}'}],
                "agentes_chamados": ["recomendacao"],
                "especialista_outputs": {
                    "recomendacao": EspecialistaOutput(conteudo="Coop Top 1", fonte="recomendacao")
                },
            }
        ],
    )

    spec_coleta = RegistroEspecialista(
        rota="coleta",
        node_name="coleta_agent",
        agente=agente_coleta,
        repositories=[],
    )
    spec_pedidos = RegistroEspecialista(
        rota="pedidos",
        node_name="recomendacao_agent",
        agente=agente_pedidos,
        repositories=[],
    )

    coop_server = A2ACooperativaServer(specialists=[spec_coleta], repositories={})
    emp_server = A2AEmpresaServer(specialists=[spec_pedidos], repositories={})
    a2a_client = A2AClient(cooperativa_server=coop_server, empresa_server=emp_server)

    router = StubAgent(
        "router",
        [
            {"messages": [], "proximos_agentes": ["coleta"], "agentes_chamados": ["router"]},
            {"messages": [], "proximos_agentes": ["pedidos"], "agentes_chamados": ["router"]},
        ],
    )
    orchestrator = StubAgent(
        "orchestrator",
        [
            {"messages": [{"role": "assistant", "content": "Resposta Coleta Formatada"}], "agentes_chamados": ["orchestrator"]},
            {"messages": [{"role": "assistant", "content": "Resposta Pedidos Formatada"}], "agentes_chamados": ["orchestrator"]},
        ],
    )

    host_graph = GraphBuilder(
        router_agent=router,
        orchestrator_agent=orchestrator,
        specialists=[spec_coleta, spec_pedidos],
        repositories={},
        checkpointer=MemorySaver(),
        guardrail_entrada=StubAgent("guardrail_entrada", [{"passou_guardrail": True}]),
        guardrail_saida=StubAgent("guardrail_saida", [{"agentes_chamados": ["guardrail_saida:ok"]}]),
        a2a_client=a2a_client,
        judge_agent=StubAgent("judge", [{"passou_juiz": True, "agentes_chamados": ["judge:apoiada noul=0.95"]}]),
    ).build_graph()

    # 1. Turno com cooperativa_id -> Host Graph despacha 'coleta' para A2ACooperativaServer
    ctx_coop = PerfilContext.from_request(cooperativa_id=str(coop_uuid))
    res_coop = host_graph.invoke(
        {
            "messages": [HumanMessage(content="Quantas coletas tivemos hoje?")],
            "cooperativa_id": ctx_coop.cooperativa_id,
            "empresa_id": None,
            "perfil_ctx": ctx_coop,
        },
        config={"configurable": {"thread_id": "sess-coop", "cooperativa_id": str(coop_uuid)}},
    )
    assert res_coop["messages"][-1].content == "Resposta Coleta Formatada"
    assert "coleta" in res_coop["agentes_chamados"]
    assert len(coop_server._tasks) == 1
    assert len(emp_server._tasks) == 0

    # 2. Turno com empresa_id -> Host Graph despacha 'pedidos' para A2AEmpresaServer
    ctx_emp = PerfilContext.from_request(empresa_id=str(emp_uuid))
    res_emp = host_graph.invoke(
        {
            "messages": [HumanMessage(content="Quais cooperativas vendem PET?")],
            "cooperativa_id": None,
            "empresa_id": ctx_emp.empresa_id,
            "perfil_ctx": ctx_emp,
        },
        config={"configurable": {"thread_id": "sess-emp", "empresa_id": str(emp_uuid)}},
    )
    assert res_emp["messages"][-1].content == "Resposta Pedidos Formatada"
    assert "recomendacao" in res_emp["agentes_chamados"]
    assert len(emp_server._tasks) == 1
