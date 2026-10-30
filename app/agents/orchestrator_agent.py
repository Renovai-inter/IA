import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState


class OrchestratorAgent(BaseAgent):
    nome_agente: str = 'orchestrator'
    descricao: str = 'Sintetizador analítico cross-agent e comunicador final das respostas dos especialistas.'

    def __init__(self, llm, system_prompt, tools=None):
        super().__init__(llm, system_prompt, tools)

    @staticmethod
    def _extrair_json_especialista(conteudo: str) -> Optional[dict[str, Any]]:
        if not conteudo or not isinstance(conteudo, str):
            return None
        texto = conteudo.strip()
        if texto.startswith('```'):
            texto = texto.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        if not (texto.startswith('{') and texto.endswith('}')):
            return None
        try:
            dados = json.loads(texto)
            return dados if isinstance(dados, dict) else None
        except (ValueError, TypeError):
            return None

    def _construir_panorama_cruzado(self, payloads_parseados: dict[str, dict[str, Any]]) -> str:
        if not payloads_parseados:
            return ''

        linhas: list[str] = [
            f'- Total de especialistas estruturados acionados: {len(payloads_parseados)} ({", ".join(payloads_parseados.keys())})'
        ]
        indicadores_globais: dict[str, Any] = {}
        janelas: dict[str, Any] = {}
        lacunas: list[str] = []
        escritas: list[str] = []

        for nome_esp, dados in payloads_parseados.items():
            ind = dados.get('indicadores')
            if isinstance(ind, dict) and ind:
                indicadores_globais[nome_esp] = ind
            jan = dados.get('janela_tempo')
            if isinstance(jan, dict) and jan:
                janelas[nome_esp] = jan
            escl = dados.get('esclarecer')
            if isinstance(escl, str) and escl.strip():
                lacunas.append(f'{nome_esp}: {escl.strip()}')
            esc = dados.get('escrita')
            if isinstance(esc, dict) and esc:
                escritas.append(f'{nome_esp}: {json.dumps(esc, ensure_ascii=False)}')

        if indicadores_globais:
            linhas.append(
                f'- Indicadores numéricos para cruzamento: {json.dumps(indicadores_globais, ensure_ascii=False)}'
            )
        if janelas:
            linhas.append(
                f'- Janelas temporais reportadas: {json.dumps(janelas, ensure_ascii=False)}'
            )
        if escritas:
            linhas.append(f'- Operações de escrita confirmadas: {"; ".join(escritas)}')
        if lacunas:
            linhas.append(f'- Lacunas / pedidos de esclarecimento prioritários: {"; ".join(lacunas)}')
        if len(payloads_parseados) > 1:
            linhas.append(
                '- Diretriz Multiagente: Cruze obrigatoriamente os indicadores e respostas acima, '
                'destacando correlações causais, trade-offs ou divergências explícitas entre os domínios.'
            )

        return '\n'.join(linhas)

    def _montar_entrada(self, state: GraphState) -> HumanMessage:
        msgs = list(state['messages'])
        pergunta = next((m for m in reversed(msgs) if self._obter_tipo_mensagem(m) in ('human', 'user')), None)
        texto_pergunta = self._obter_texto_mensagem(pergunta) if pergunta else ''

        especialista_outputs = state.get('especialista_outputs', {})
        payloads_parseados: dict[str, dict[str, Any]] = {}

        if especialista_outputs:
            blocos = []
            for nome_esp, output in especialista_outputs.items():
                fonte_val = getattr(output, 'fonte', None) or (output.get('fonte') if isinstance(output, dict) else None)
                conteudo_val = getattr(output, 'conteudo', None) or (output.get('conteudo') if isinstance(output, dict) else str(output))
                dados_json = self._extrair_json_especialista(str(conteudo_val))
                if dados_json is not None:
                    payloads_parseados[nome_esp] = dados_json
                    conteudo_fmt = json.dumps(dados_json, ensure_ascii=False, indent=2)
                else:
                    conteudo_fmt = str(conteudo_val)
                fonte_info = f' (Fonte: {fonte_val})' if fonte_val else ''
                blocos.append(
                    f'--- ESPECIALISTA: {nome_esp}{fonte_info} ---\n{conteudo_fmt}'
                )
            corpo_saidas = '\n\n'.join(blocos)
        else:
            saida_especialista = msgs[-1] if msgs else None
            corpo_saidas = self._obter_texto_mensagem(saida_especialista) if saida_especialista else ''
            dados_json = self._extrair_json_especialista(corpo_saidas)
            if dados_json is not None:
                dominio_fallback = str(dados_json.get('dominio') or 'especialista')
                payloads_parseados[dominio_fallback] = dados_json

        panorama = self._construir_panorama_cruzado(payloads_parseados)
        bloco_panorama = (
            f'PANORAMA_CRUZADO_DOS_ESPECIALISTAS:\n{panorama}\n\n'
            if panorama
            else ''
        )

        return HumanMessage(content=(
            'PERGUNTA_DO_USUARIO:\n'
            f'{texto_pergunta}\n\n'
            f'{bloco_panorama}'
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
