"""Testes do RecomendacaoAgent.run() — sem LLM real (mesma técnica de
monkeypatch de create_agent usada em test_router_agent.py)."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.recomendacao_agent as recomendacao_module
from app.agents.recomendacao_agent import RecomendacaoAgent


class _StubRunnable:
    def __init__(self, texto_resposta: str):
        self._texto = texto_resposta

    def invoke(self, inputs, config=None):
        return {'messages': [AIMessage(content=self._texto)]}


def test_recomendacao_agent_monta_especialista_output(monkeypatch: pytest.MonkeyPatch):
    texto_json = (
        '{"dominio":"recomendacao","intencao":"recomendar",'
        '"resposta":"Recomendo a Cooperativa Recicla Paulistana a 12,4 km com 3.200 kg de PET em estoque.",'
        '"recomendacao":"Abra proposta diretamente com a Cooperativa Recicla Paulistana."}'
    )
    stub = _StubRunnable(texto_json)
    monkeypatch.setattr(recomendacao_module, 'create_agent', lambda **kwargs: stub)

    agent = RecomendacaoAgent(llm=None, system_prompt='prompt de teste', tools=[])

    resultado = agent.run(
        {
            'messages': [HumanMessage(content='preciso comprar 2 toneladas de PET em SP')],
            'snapshots': {},
        },
        config={},
    )

    assert resultado['agentes_chamados'] == ['recomendacao']

    output = resultado['especialista_outputs']['recomendacao']
    assert output.conteudo == texto_json
    assert output.fonte == 'recomendacao'
    assert output.veredito_juiz == 'pendente'
