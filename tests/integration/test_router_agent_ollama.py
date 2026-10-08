"""RouterAgent rodando com um modelo Ollama local de verdade (tier LOW — o
mesmo tier usado em produção pelo router_agent, ver Settings.AGENT_LLM_MAP).

Isso valida se o protocolo ROUTE=... é seguido por um modelo pequeno o
suficiente pra rodar localmente. NÃO é garantido que um modelo local de 3B
seja tão preciso quanto o gpt-oss-20b (Groq) usado em produção — se os
casos abaixo falharem de forma consistente, é sinal de que o tier LOW
local precisa de um modelo maior (troque em app/llms/ollama_provider.py),
não necessariamente um bug no RouterAgent.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agents.router_agent import RouterAgent
from app.prompts import ROUTER_PROMPT_COMPLETO

from tests.integration.ollama_helpers import check_modelo_ou_skip, requires_ollama

CASOS = [
    ("quanto de papelão temos disponível no estoque?", "estoque"),
    ("quais são as políticas de privacidade do Renovaí?", "rag_faq"),
    ("oi, bom dia", "fim"),
]


@requires_ollama
@pytest.mark.parametrize("pergunta, rota_esperada", CASOS)
def test_router_agent_classifica_com_ollama(ollama_llm_factory, pergunta, rota_esperada):
    llm = ollama_llm_factory.get("OLLAMA", "LOW")
    check_modelo_ou_skip(llm, tier="LOW")
    agent = RouterAgent(llm=llm, system_prompt=ROUTER_PROMPT_COMPLETO, tools=[])

    resultado = agent.run({"messages": [HumanMessage(content=pergunta)]}, config={})

    assert resultado["proximo_agente"] == rota_esperada, (
        f"modelo local classificou {pergunta!r} como "
        f"{resultado['proximo_agente']!r}, esperado {rota_esperada!r} "
        f"— resposta crua do modelo: {resultado['messages'][0]['content']!r}"
    )