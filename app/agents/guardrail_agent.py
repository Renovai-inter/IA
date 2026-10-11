"""
Agente Guardrail: 1 agente, 2 nós no grafo.

A lógica de política mora em app/guardrails/ (funções puras); aqui fica só a
orquestração: determinístico primeiro, LLM depois, e a tradução para o GraphState.
"""
from __future__ import annotations

import logging
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState
from app.guardrails import input_guardrails as entrada
from app.guardrails import output_guardrails as saida

log = logging.getLogger('guardrail')


def _ultima_humana(state: GraphState):
    return next((m for m in reversed(list(state['messages'])) if m.type == 'human'), None)


class GuardrailEntradaAgent(BaseAgent):
    nome_agente: str = 'guardrail_entrada'
    descricao: str = 'Bloqueia injeção, acesso a dados internos e conteúdo fora da política.'

    def _processar_resposta_llm(self, bruto: str):
        categoria = entrada.interpretar_classificacao(bruto)
        if 'CATEGORIA' not in bruto:
            log.warning('guardrail_entrada: saída ilegível do classificador, fail-open: %r', bruto[:200])
        return entrada.resultado_da_categoria(categoria)

    def _montar_update(self, resultado) -> dict:
        update = {
            'passou_guardrail': not resultado.bloqueado,
            'motivo_guardrail': resultado.motivo,
            'agentes_chamados': [f'{self.nome_agente}:{resultado.motivo}'],
        }
        if resultado.bloqueado:
            update['messages'] = [{'role': 'assistant', 'content': resultado.mensagem}]
        return update

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        pergunta = _ultima_humana(state)
        texto = self._obter_texto_mensagem(pergunta) if pergunta else ''

        resultado = entrada.verificar_deterministico(texto)

        if resultado is None:
            perfil = state.get('perfil_ctx')
            tipo_perfil = perfil.tipo if perfil else 'DESCONHECIDO'
            resposta = self.llm.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=(
                        f'TIPO_PERFIL: {tipo_perfil}\n'
                        f'<mensagem_usuario>\n{texto}\n</mensagem_usuario>'
                    )),
                ],
                config=config,
            )
            bruto = self._obter_texto_mensagem(resposta)
            resultado = self._processar_resposta_llm(bruto)

        return self._montar_update(resultado)

    async def arun(self, state: GraphState, config: RunnableConfig) -> dict:
        pergunta = _ultima_humana(state)
        texto = self._obter_texto_mensagem(pergunta) if pergunta else ''

        resultado = entrada.verificar_deterministico(texto)

        if resultado is None:
            perfil = state.get('perfil_ctx')
            tipo_perfil = perfil.tipo if perfil else 'DESCONHECIDO'
            resposta = await self.llm.ainvoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=(
                        f'TIPO_PERFIL: {tipo_perfil}\n'
                        f'<mensagem_usuario>\n{texto}\n</mensagem_usuario>'
                    )),
                ],
                config=config,
            )
            bruto = self._obter_texto_mensagem(resposta)
            resultado = self._processar_resposta_llm(bruto)

        return self._montar_update(resultado)


class GuardrailSaidaAgent(BaseAgent):
    nome_agente: str = 'guardrail_saida'
    descricao: str = 'Higieniza a resposta final (PII, protocolo interno) e aplica política por rota.'

    def __init__(self, llm, system_prompt, tools=None, rotas_revisao: Iterable[str] = ()):
        super().__init__(llm, system_prompt, tools)
        self._rotas_revisao = set(rotas_revisao)

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        mapa_pii = ((config or {}).get('configurable') or {}).get('mapa_pii') or {}
        ultima = list(state['messages'])[-1]

        if ultima.type != 'ai':
            return {'agentes_chamados': [f'{self.nome_agente}:sem_resposta']}

        original = self._obter_texto_mensagem(ultima)

        texto, motivo = saida.higienizar(original, mapa_pii)

        rotas_ativas = state.get('proximos_agentes') or []
        revisar = '*' in self._rotas_revisao or any(r in self._rotas_revisao for r in rotas_ativas)
        if revisar and motivo != 'vazamento_protocolo':
            resposta = self.llm.invoke(
                [SystemMessage(content=self.system_prompt),
                 HumanMessage(content=f'<resposta>\n{texto}\n</resposta>')],
                config=config,
            )
            veredito = saida.interpretar_veredito(self._obter_texto_mensagem(resposta))
            texto, motivo_llm = saida.aplicar_veredito(texto, veredito)
            motivo = motivo_llm or motivo

        update = {'agentes_chamados': [f'{self.nome_agente}:{motivo or 'ok'}']}
        if texto != original:
            update['messages'] = [{'role': 'assistant', 'content': texto, 'id': ultima.id}]
        return update

    async def arun(self, state: GraphState, config: RunnableConfig) -> dict:
        mapa_pii = ((config or {}).get('configurable') or {}).get('mapa_pii') or {}
        ultima = list(state['messages'])[-1]

        if ultima.type != 'ai':
            return {'agentes_chamados': [f'{self.nome_agente}:sem_resposta']}

        original = self._obter_texto_mensagem(ultima)

        texto, motivo = saida.higienizar(original, mapa_pii)

        rotas_ativas = state.get('proximos_agentes') or []
        revisar = '*' in self._rotas_revisao or any(r in self._rotas_revisao for r in rotas_ativas)
        if revisar and motivo != 'vazamento_protocolo':
            resposta = await self.llm.ainvoke(
                [SystemMessage(content=self.system_prompt),
                 HumanMessage(content=f'<resposta>\n{texto}\n</resposta>')],
                config=config,
            )
            veredito = saida.interpretar_veredito(self._obter_texto_mensagem(resposta))
            texto, motivo_llm = saida.aplicar_veredito(texto, veredito)
            motivo = motivo_llm or motivo

        update = {'agentes_chamados': [f'{self.nome_agente}:{motivo or 'ok'}']}
        if texto != original:
            update['messages'] = [{'role': 'assistant', 'content': texto, 'id': ultima.id}]
        return update
