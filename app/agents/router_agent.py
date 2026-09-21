from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState

from langchain.agents import create_agent

class RouterAgent(BaseAgent):
    nome_agente: str = 'router'
    descricao: str = ''

    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

        self._runnable = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
        )

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        resultado = self._runnable.invoke(
            {'messages': list(state['messages'])},
            config=(config or {}).get('configurable', {})
            )
        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])

        rota = 'fim'
        for linha in texto_resposta.splitlines():
            if linha.startswith('ROUTE='):
                rota = linha.split('=', 1)[1].strip()
                break

        return {
            'messages'        : [{'role': 'assistant', 'content': texto_resposta}],
            'proximo_agente'  : rota,
            'agentes_chamados': [self.nome_agente],
        }