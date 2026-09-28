"""Testes do RouterAgent.run() — isolados de qualquer LLM real.

Técnica: monkeypatch de `create_agent` (tanto em app.agents.base quanto em
app.agents.router_agent, que importa e chama de novo no __init__) para
devolver um runnable-dublê com resposta fixa. Isso evita qualquer
dependência de bind_tools/tool-calling da versão instalada do LangChain —
testamos só a lógica de parsing do protocolo ROUTE=..., que é o que
realmente pertence ao RouterAgent.

Também documenta o bug ativo conhecido do roteador (ver a skill
ia-renovai / estado_atual_codigo.md): o texto cru do protocolo
ROUTE=.../PERGUNTA_ORIGINAL=... é sempre adicionado ao histórico de
mensagens, mesmo quando a rota não é 'fim'. Isso não foi corrigido aqui —
só documentado via teste xfail, pra virar verde sozinho quando alguém
aplicar a correção mínima descrita no arquivo de referência.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.base as base_module
import app.agents.router_agent as router_agent_module
from app.agents.router_agent import RouterAgent


class _StubRunnable:
    def __init__(self, texto_resposta: str):
        self._texto = texto_resposta

    def invoke(self, inputs, config=None):
        return {"messages": [AIMessage(content=self._texto)]}


def _build_router_agent(monkeypatch: pytest.MonkeyPatch, texto_resposta: str) -> RouterAgent:
    stub = _StubRunnable(texto_resposta)
    monkeypatch.setattr(base_module, "create_agent", lambda **kwargs: stub)
    monkeypatch.setattr(router_agent_module, "create_agent", lambda **kwargs: stub)
    return RouterAgent(llm=None, system_prompt="prompt de teste", tools=[])


def test_router_agent_extrai_rota_do_protocolo(monkeypatch: pytest.MonkeyPatch):
    texto = "ROUTE=estoque\nPERGUNTA_ORIGINAL=quanto de papelão temos?"
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="quanto de papelão temos?")]},
        config={},
    )

    assert resultado["proximo_agente"] == "estoque"
    assert resultado["agentes_chamados"] == ["router"]


def test_router_agent_sem_protocolo_cai_em_fim(monkeypatch: pytest.MonkeyPatch):
    texto = (
        "Olá! Posso te ajudar com estoque, coletas, pedidos, financeiro, "
        "rotas ou dúvidas sobre o Renovaí."
    )
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="oi")]},
        config={},
    )

    assert resultado["proximo_agente"] == "fim"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug conhecido do RouterAgent (ver estado_atual_codigo.md): o protocolo "
        "ROUTE=.../PERGUNTA_ORIGINAL=... cru é sempre adicionado ao histórico de "
        "mensagens, mesmo quando a rota != 'fim'. Correção mínima: só incluir a "
        "mensagem do roteador no histórico quando ele está respondendo diretamente "
        "ao usuário (rota == 'fim'). Este teste passa sozinho quando isso for corrigido."
    ),
)
def test_router_agent_nao_deveria_vazar_protocolo_quando_encaminha(monkeypatch: pytest.MonkeyPatch):
    texto = "ROUTE=estoque\nPERGUNTA_ORIGINAL=quanto de papelão temos?"
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="quanto de papelão temos?")]},
        config={},
    )

    mensagem_no_historico = resultado["messages"][0]["content"]
    assert "ROUTE=" not in mensagem_no_historico