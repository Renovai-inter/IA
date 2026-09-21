from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState

from langchain.agents import create_agent

class OrchestratorAgent(BaseAgent):
    nome_agente: str = 'orchestrator'
    descricao: str = ''

    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

        self._runnable = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
        )

    def run(self, state: GraphState, config: RunnableConfig):
        resultado = self._runnable.invoke(
            {'messages': list(state['messages'])},
            config=(config or {}).get('configurable', {})
        )
        
        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])
        
        return {
            'messages'        : [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados': [self.nome_agente],
        }