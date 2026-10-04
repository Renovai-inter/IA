from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import EspecialistaOutput, GraphState


class FaqAgent(BaseAgent):
    nome_agente: str = 'faq'
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
            config=(config or {}).get('configurable', {})
            )

        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])
        faq_output = EspecialistaOutput(
            conteudo=texto_resposta,
            fonte=self.nome_agente,
        )

        return {
            'messages'            : [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados'    : [self.nome_agente],
            'especialista_outputs': {self.nome_agente: faq_output},
        }