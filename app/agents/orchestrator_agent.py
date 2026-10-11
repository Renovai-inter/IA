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
        texto_pergunta = self._obter_texto_mensagem(pergunta) if pergunta else ''

        especialista_outputs = state.get('especialista_outputs', {})

        if especialista_outputs:
            blocos = []
            for nome_esp, output in especialista_outputs.items():
                fonte_info = f" (Fonte: {output.fonte})" if output.fonte else ""
                blocos.append(
                    f"--- ESPECIALISTA: {nome_esp}{fonte_info} ---\n{output.conteudo}"
                )
            corpo_saidas = "\n\n".join(blocos)
        else:
            saida_especialista = msgs[-1] if msgs else None
            corpo_saidas = self._obter_texto_mensagem(saida_especialista) if saida_especialista else ''

        entrada = HumanMessage(content=(
            "PERGUNTA_DO_USUARIO:\n"
            f"{texto_pergunta}\n\n"
            "SAIDAS_DOS_ESPECIALISTAS:\n"
            f"{corpo_saidas}"
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