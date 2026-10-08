"""Testes do MaterialEstoqueAgent.run() — sem LLM real (mesma técnica de
monkeypatch de create_agent usada em test_router_agent.py)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.material_estoque_agent as material_estoque_module
from app.agents.material_estoque_agent import MaterialEstoqueAgent


class _StubRunnable:
    def __init__(self, texto_resposta: str):
        self._texto = texto_resposta

    def invoke(self, inputs, config=None):
        return {"messages": [AIMessage(content=self._texto)]}


def test_material_estoque_agent_monta_especialista_output(monkeypatch: pytest.MonkeyPatch):
    texto_json = (
        '{"dominio":"material_estoque","intencao":"consultar",'
        '"resposta":"Há 3.450 kg de papelão disponível."}'
    )
    stub = _StubRunnable(texto_json)
    monkeypatch.setattr(material_estoque_module, "create_agent", lambda **kwargs: stub)

    agent = MaterialEstoqueAgent(llm=None, system_prompt="prompt de teste", tools=[])

    resultado = agent.run(
        {
            "messages": [HumanMessage(content="quanto de papelão temos?")],
            "snapshots": {"material_snapshot": SimpleNamespace(itens=[])},
        },
        config={},
    )

    assert resultado["agentes_chamados"] == ["material_estoque"]

    output = resultado["especialista_outputs"]["material_estoque"]
    assert output.conteudo == texto_json
    assert output.fonte == "material_estoque"
    assert output.veredito_juiz == "pendente"