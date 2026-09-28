"""Dublês (test doubles) compartilhados entre tests/unit e tests/integration.

Não é um arquivo de teste (não começa com test_*, o pytest não o coleta) —
é importado explicitamente com `from tests.support import ...`.
"""
from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent


class StubAgent(BaseAgent):
    """Agente-dublê: nunca chama create_agent nem nenhum LLM real.

    Serve para testar roteamento/grafo (GraphBuilder) isolado do LangChain —
    cada chamada a `run()` devolve a próxima resposta da lista `respostas`
    (a última é reutilizada se `run()` for chamado mais vezes do que há
    respostas cadastradas).
    """

    def __init__(self, nome_agente: str, respostas: list[dict[str, Any]]):
        self.nome_agente = nome_agente
        self.llm = None
        self.system_prompt = ""
        self.tools: list = []
        self._respostas = list(respostas)
        self._chamadas = 0

    def run(self, state: dict, config: RunnableConfig | None = None) -> dict:
        self._chamadas += 1
        indice = min(self._chamadas - 1, len(self._respostas) - 1)
        return self._respostas[indice]


class StubRepository:
    """Repository-dublê: devolve sempre o mesmo snapshot, sem tocar em Postgres.

    Satisfaz o único contrato que `make_repo_backed_node` (app/graph/nodes.py)
    realmente usa: `get_snapshot(cooperativa_id) -> Any`.
    """

    def __init__(self, snapshot: Any):
        self._snapshot = snapshot

    def get_snapshot(self, cooperativa_id) -> Any:
        return self._snapshot