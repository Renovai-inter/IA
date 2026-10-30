from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent

from app.agents.base import BaseAgent
from app.graph.state import GraphState, EspecialistaOutput


class MaterialEstoqueAgent(BaseAgent):
    nome_agente: str = 'material_estoque'
    descricao: str = 'Especialista em estoque, materiais recicláveis e movimentações.'

    def __init__(self, llm, system_prompt, tools=None):
        super().__init__(llm, system_prompt, tools)

        self._runnable = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
        )

    def _processar_resultado(self, resultado: dict) -> dict:
        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])
        material_estoque_output = EspecialistaOutput(
            conteudo=texto_resposta,
            fonte=self.nome_agente,
        )

        return {
            'messages': [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados': [self.nome_agente],
            'especialista_outputs': {self.nome_agente: material_estoque_output},
        }

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        resultado = self._runnable.invoke(
            {'messages': self._mensagens_recentes(state)},
            config=self._config_interno(config, snapshots=state.get('snapshots') or {})
        )
        return self._processar_resultado(resultado)

    async def arun(self, state: GraphState, config: RunnableConfig) -> dict:
        resultado = await self._runnable.ainvoke(
            {'messages': self._mensagens_recentes(state)},
            config=self._config_interno(config, snapshots=state.get('snapshots') or {})
        )
        return self._processar_resultado(resultado)
