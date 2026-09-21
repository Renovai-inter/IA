from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState, EspecialistaOutput

from langchain.agents import create_agent

class MaterialEstoqueAgent(BaseAgent):
    nome_agente: str = 'nome_agente'
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
            config=(config or {}).get('configurable', {}) | {'snapshots': state['snapshots']}
        )

        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])
        material_estoque_output = EspecialistaOutput(
            conteudo=texto_resposta,
            fonte=self.nome_agente,
        )

        return {
            'messages'            : [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados'    : [self.nome_agente],
            'especialista_outputs': {self.nome_agente: material_estoque_output},
        }