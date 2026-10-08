from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState


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
            {'messages': self._mensagens_recentes(state)},
            config=self._config_interno(config)
        )

        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])

        rota = 'fim'
        for linha in texto_resposta.splitlines():
            if linha.startswith('ROUTE='):
                rota = linha.split('=', 1)[1].strip()
                break

        # O protocolo ROUTE=.../PERGUNTA_ORIGINAL=... é comunicação interna
        # roteador -> grafo: só vira mensagem no histórico quando o roteador
        # está respondendo direto ao usuário (saudação / fora de escopo).
        # Quando encaminha, o especialista lê a pergunta do próprio
        # HumanMessage, e a rota segue em `proximo_agente`.
        mensagens = (
            [{'role': 'assistant', 'content': texto_resposta}]
            if rota == 'fim' else []
        )

        return {
            'messages'        : mensagens,
            'proximo_agente'  : rota,
            'agentes_chamados': [self.nome_agente],
        }