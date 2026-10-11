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

        rotas: list[str] = []
        for linha in texto_resposta.splitlines():
            linha_strip = linha.strip()
            if linha_strip.startswith('ROUTES='):
                payload = linha_strip.split('=', 1)[1].strip().strip('[]')
                rotas = [r.strip().strip("'\"") for r in payload.split(',') if r.strip()]
                break
            elif linha_strip.startswith('ROUTE='):
                payload = linha_strip.split('=', 1)[1].strip().strip('[]')
                rotas = [r.strip().strip("'\"") for r in payload.split(',') if r.strip()]
                break

        rotas = [r for r in rotas if r not in ('fim', 'fim_direto')]
        eh_fim = len(rotas) == 0

        # O protocolo ROUTES=.../PERGUNTA_ORIGINAL=... é comunicação interna
        # roteador -> grafo: só vira mensagem no histórico quando o roteador
        # está respondendo direto ao usuário (saudação / fora de escopo).
        # Quando encaminha, os especialistas leem a pergunta do próprio
        # HumanMessage, e as rotas seguem em `proximos_agentes`.
        mensagens = (
            [{'role': 'assistant', 'content': texto_resposta}]
            if eh_fim else []
        )

        return {
            'messages'        : mensagens,
            'proximos_agentes': rotas,
            'agentes_chamados': [self.nome_agente],
        }