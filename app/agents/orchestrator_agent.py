from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState


class OrchestratorAgent(BaseAgent):
    nome_agente: str = 'orchestrator'
    descricao: str = 'Consolidador final das respostas dos especialistas operacionais.'

    def __init__(self, llm, system_prompt, tools=None):
        super().__init__(llm, system_prompt, tools)

    def _montar_entrada(self, state: GraphState) -> HumanMessage:
        msgs = list(state['messages'])
        pergunta = next((m for m in reversed(msgs) if self._obter_tipo_mensagem(m) in ('human', 'user')), None)
        texto_pergunta = self._obter_texto_mensagem(pergunta) if pergunta else ''

        especialista_outputs = state.get('especialista_outputs', {})

        if especialista_outputs:
            blocos = []
            for nome_esp, output in especialista_outputs.items():
                fonte_val = getattr(output, 'fonte', None) or (output.get('fonte') if isinstance(output, dict) else None)
                conteudo_val = getattr(output, 'conteudo', None) or (output.get('conteudo') if isinstance(output, dict) else str(output))
                fonte_info = f' (Fonte: {fonte_val})' if fonte_val else ''
                blocos.append(
                    f'--- ESPECIALISTA: {nome_esp}{fonte_info} ---\n{conteudo_val}'
                )
            corpo_saidas = '\n\n'.join(blocos)
        else:
            saida_especialista = msgs[-1] if msgs else None
            corpo_saidas = self._obter_texto_mensagem(saida_especialista) if saida_especialista else ''

        return HumanMessage(content=(
            'PERGUNTA_DO_USUARIO:\n'
            f'{texto_pergunta}\n\n'
            'SAIDAS_DOS_ESPECIALISTAS:\n'
            f'{corpo_saidas}'
        ))

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        entrada = self._montar_entrada(state)
        resultado = self.llm.invoke(
            [SystemMessage(content=self.system_prompt), entrada],
            config=config,
        )
        texto_resposta = self._obter_texto_mensagem(resultado)
        return {
            'messages': [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados': [self.nome_agente],
        }

    async def arun(self, state: GraphState, config: RunnableConfig) -> dict:
        entrada = self._montar_entrada(state)
        resultado = await self.llm.ainvoke(
            [SystemMessage(content=self.system_prompt), entrada],
            config=config,
        )
        texto_resposta = self._obter_texto_mensagem(resultado)
        return {
            'messages': [{'role': 'assistant', 'content': texto_resposta}],
            'agentes_chamados': [self.nome_agente],
        }
