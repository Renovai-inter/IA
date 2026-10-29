"""Testes unitários do FinanceiroAgent — sem LLM real via monkeypatch."""
from __future__ import annotations

from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.financeiro_agent as financeiro_module
from app.agents.financeiro_agent import FinanceiroAgent


class _StubRunnable:
    def __init__(self, texto_resposta: str):
        self._texto = texto_resposta

    def invoke(self, inputs, config=None):
        return {'messages': [AIMessage(content=self._texto)]}

    async def ainvoke(self, inputs, config=None):
        return {'messages': [AIMessage(content=self._texto)]}


def test_financeiro_agent_monta_especialista_output(monkeypatch: pytest.MonkeyPatch):
    texto_json = (
        '{"dominio":"financeiro","intencao":"consultar",'
        '"resposta":"O rateio de setembro/2026 totalizou R$ 26.400,00 líquidos."}'
    )
    stub = _StubRunnable(texto_json)
    monkeypatch.setattr(financeiro_module, 'create_agent', lambda **kwargs: stub)

    agent = FinanceiroAgent(llm=None, system_prompt='prompt de teste', tools=[])

    resultado = agent.run(
        {
            'messages': [HumanMessage(content='qual o rateio de setembro?')],
            'snapshots': {
                'rateio_snapshot': SimpleNamespace(itens=[]),
                'rateio_funcionario_snapshot': SimpleNamespace(itens=[]),
                'despesa_snapshot': SimpleNamespace(itens=[]),
            },
        },
        config={},
    )

    assert resultado['agentes_chamados'] == ['financeiro']
    output = resultado['especialista_outputs']['financeiro']
    assert output.conteudo == texto_json
    assert output.fonte == 'financeiro'


@pytest.mark.asyncio
async def test_financeiro_agent_arun(monkeypatch: pytest.MonkeyPatch):
    texto_json = '{"dominio":"financeiro","intencao":"consultar","resposta":"Despesas de R$ 5.000,00."}'
    stub = _StubRunnable(texto_json)
    monkeypatch.setattr(financeiro_module, 'create_agent', lambda **kwargs: stub)

    agent = FinanceiroAgent(llm=None, system_prompt='prompt de teste', tools=[])

    resultado = await agent.arun(
        {
            'messages': [HumanMessage(content='quais as despesas?')],
            'snapshots': {
                'rateio_snapshot': SimpleNamespace(itens=[]),
                'rateio_funcionario_snapshot': SimpleNamespace(itens=[]),
                'despesa_snapshot': SimpleNamespace(itens=[]),
            },
        },
        config={},
    )

    assert resultado['agentes_chamados'] == ['financeiro']
    assert resultado['especialista_outputs']['financeiro'].conteudo == texto_json
