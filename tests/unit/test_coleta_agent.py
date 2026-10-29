"""Testes unitários do ColetaAgent — sem LLM real via monkeypatch."""
from __future__ import annotations

from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.coleta_agent as coleta_module
from app.agents.coleta_agent import ColetaAgent


class _StubRunnable:
    def __init__(self, texto_resposta: str):
        self._texto = texto_resposta

    def invoke(self, inputs, config=None):
        return {'messages': [AIMessage(content=self._texto)]}

    async def ainvoke(self, inputs, config=None):
        return {'messages': [AIMessage(content=self._texto)]}


def test_coleta_agent_monta_especialista_output(monkeypatch: pytest.MonkeyPatch):
    texto_json = (
        '{"dominio":"coleta","intencao":"consultar",'
        '"resposta":"Há 2 coletas agendadas para hoje."}'
    )
    stub = _StubRunnable(texto_json)
    monkeypatch.setattr(coleta_module, 'create_agent', lambda **kwargs: stub)

    agent = ColetaAgent(llm=None, system_prompt='prompt de teste', tools=[])

    resultado = agent.run(
        {
            'messages': [HumanMessage(content='quais as coletas de hoje?')],
            'snapshots': {
                'coleta_snapshot': SimpleNamespace(itens=[]),
                'triagem_snapshot': SimpleNamespace(itens=[]),
            },
        },
        config={},
    )

    assert resultado['agentes_chamados'] == ['coleta']
    output = resultado['especialista_outputs']['coleta']
    assert output.conteudo == texto_json
    assert output.fonte == 'coleta'


@pytest.mark.asyncio
async def test_coleta_agent_arun(monkeypatch: pytest.MonkeyPatch):
    texto_json = '{"dominio":"coleta","intencao":"status","resposta":"Coleta COL-1 concluída."}'
    stub = _StubRunnable(texto_json)
    monkeypatch.setattr(coleta_module, 'create_agent', lambda **kwargs: stub)

    agent = ColetaAgent(llm=None, system_prompt='prompt de teste', tools=[])

    resultado = await agent.arun(
        {
            'messages': [HumanMessage(content='status da coleta COL-1')],
            'snapshots': {
                'coleta_snapshot': SimpleNamespace(itens=[]),
                'triagem_snapshot': SimpleNamespace(itens=[]),
            },
        },
        config={},
    )

    assert resultado['agentes_chamados'] == ['coleta']
    assert resultado['especialista_outputs']['coleta'].conteudo == texto_json
