from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage

from app.agents.base import BaseAgent
from app.graph.state import GraphState

class OrchestratorAgent(BaseAgent):
    nome_agente: str = 'orchestrator'
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
        
        texto_resposta = self._obter_texto_mensagem(resultado)
        
        return {
            'messages'        : [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados': [self.nome_agente],
        }