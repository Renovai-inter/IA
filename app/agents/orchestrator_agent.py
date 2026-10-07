from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.base import BaseAgent
from app.graph.state import GraphState

class OrchestratorAgent(BaseAgent):
    nome_agente: str = 'orchestrator'
    descricao: str = ''

    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        msgs = list(state['messages'])
        pergunta = next((m for m in reversed(msgs) if m.type == 'human'), None)
        saida_especialista = msgs[-1]

        entrada = HumanMessage(content=(
            "PERGUNTA_DO_USUARIO:\n"
            f"{self._obter_texto_mensagem(pergunta) if pergunta else ''}\n\n"
            "SAIDA_DO_ESPECIALISTA:\n"
            f"{self._obter_texto_mensagem(saida_especialista)}"
        ))

        resultado = self.llm.invoke(
            [SystemMessage(content=self.system_prompt), entrada],
            config=config,
        )
        
        texto_resposta = self._obter_texto_mensagem(resultado)
        
        return {
            'messages'        : [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados': [self.nome_agente],
        }