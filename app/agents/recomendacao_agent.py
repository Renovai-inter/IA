from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState, EspecialistaOutput, resolver_contexto_dominio

from langchain.agents import create_agent


class RecomendacaoAgent(BaseAgent):
    nome_agente: str = 'recomendacao'
    descricao: str = ''

    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

        self._runnable = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
        )

    def _processar_resultado(self, resultado: dict) -> dict:
        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])
        recomendacao_output = EspecialistaOutput(
            conteudo=texto_resposta,
            fonte=self.nome_agente,
        )

        return {
            'messages'            : [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados'    : [self.nome_agente],
            'especialista_outputs': {self.nome_agente: recomendacao_output},
        }

    def _montar_config_execucao(self, state: GraphState, config: RunnableConfig | None) -> RunnableConfig:
        _, empresa_id, _ = resolver_contexto_dominio(state, config)
        extras = {'snapshots': state.get('snapshots') or {}}
        if empresa_id is not None:
            extras['empresa_id'] = str(empresa_id)
        return self._config_interno(config, **extras)

    def run(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        resultado = self._runnable.invoke(
            {'messages': self._mensagens_recentes(state)},
            config=self._montar_config_execucao(state, config),
        )
        return self._processar_resultado(resultado)

    async def arun(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        resultado = await self._runnable.ainvoke(
            {'messages': self._mensagens_recentes(state)},
            config=self._montar_config_execucao(state, config),
        )
        return self._processar_resultado(resultado)
