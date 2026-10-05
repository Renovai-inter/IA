from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState


class RouterAgent(BaseAgent):
    nome_agente: str = 'router'
    descricao: str = ''

    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        resultado = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                *self._mensagens_recentes(state),
            ],
            config=(config or {}).get('configurable', {}),
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