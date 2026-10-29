"""Agente Juiz (JudgeAgent) — Controle de Alucinação e Verificação de Evidências.

Implementa um modelo de decisão discriminativo ('System One') no portal de saída do grafo.
Avalia a resposta final (orquestrada/sintetizada) contra as evidências factuais do turno
(snapshots de banco, saídas de especialistas e chunks RAG do Qdrant/PNRS).

Não redige texto livre nem reinjeta ciclos lentos de reflexão.
Se a resposta for fiel (choice == 'apoiada' e noul >= limiar), o texto original segue intacto.
Se a resposta for infundada ou extrapolar a evidência, ela é substituída de forma
determinística por uma frase segura pré-definida.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import EspecialistaOutput, GraphState

log = logging.getLogger('renovai.judge')

FRASE_SEGURA = (
    'Não localizei evidências suficientes nos registros operacionais da cooperativa '
    'ou nas normas vigentes para confirmar essas informações com exatidão.'
)
LIMIAR_NOUL_PADRAO = 0.70

REGRAS_AVALIACAO = (
    'A pergunta do usuário é evidência do que ele solicitou, declarou ou informou ter feito. '
    'Somas, contagens, conclusões lógicas, sugestões de próximo passo, ofertas de ajuda, '
    'saudações e confirmação de ausência de registros contam como apoiada. '
    'Valores numéricos de estoque (kg), preços médios, custos, despesas, datas de coleta, '
    'triagens, rotas de veículos, nomes de cooperativas e artigos da lei PNRS devem '
    'estar estritamente fundamentados na evidência disponível. '
    'Números inventados, moedas trocadas, valores sem evidência ou afirmações factuais '
    'não suportadas contam como sem_apoio.'
)


class DecisionEngine:
    """Interface abstrata para o motor de decisão do Juiz."""
    def predict(self, state: dict, questions: list[dict]) -> dict:
        raise NotImplementedError


class LayaDecisionEngine(DecisionEngine):
    """Motor de decisão usando o modelo open source Laya (Convai Innovations)."""
    def __init__(self, model_checkpoint: str = 'convaiinnovations/laya-multilingual'):
        from laya import Router
        self._router = Router(model_checkpoint=model_checkpoint)

    def predict(self, state: dict, questions: list[dict]) -> dict:
        return self._router.predict(
            state=json.dumps(state, ensure_ascii=False),
            questions=questions,
        )


def _normalizar_numero(num_str: str) -> set[str]:
    variantes = {num_str}
    limpo = num_str.replace('.', '')
    variantes.add(limpo)
    if ',' in limpo:
        com_ponto = limpo.replace(',', '.')
        variantes.add(com_ponto)
        try:
            val = float(com_ponto)
            if val.is_integer():
                variantes.add(str(int(val)))
        except ValueError:
            pass
    elif '.' in num_str:
        try:
            val = float(num_str)
            if val.is_integer():
                variantes.add(str(int(val)))
        except ValueError:
            pass
    return variantes


class DeterministicDecisionEngine(DecisionEngine):
    """Motor de decisão determinístico baseado em evidências e regras semânticas.
    Atua em ambientes sem Laya instalado / GPU / offline, garantindo a mesma interface
    e saída tipada de 'choice' e 'noul' calibrado."""

    def predict(
        self,
        state_or_pergunta: dict | str,
        questions_or_resposta: list[dict] | str | None = None,
        evidencia: str = '',
    ) -> dict:
        if isinstance(state_or_pergunta, dict):
            pergunta = state_or_pergunta.get('pergunta', '')
            resposta = state_or_pergunta.get('resposta', '')
            evidencia = state_or_pergunta.get('evidencia', '')
        else:
            pergunta = state_or_pergunta or ''
            resposta = questions_or_resposta if isinstance(questions_or_resposta, str) else ''
            evidencia = evidencia or ''

        choice, noul = self._avaliar(pergunta, resposta, evidencia)

        probs = {
            'apoiada': 0.88 if choice == 'apoiada' else (0.30 if choice == 'parcial' else 0.08),
            'parcial': 0.55 if choice == 'parcial' else 0.10,
            'sem_apoio': 0.85 if choice == 'sem_apoio' else (0.15 if choice == 'parcial' else 0.02),
        }
        return {
            'veredito': {
                'type': 'choice',
                'choice': choice,
                'probabilities': probs,
                'confidence': max(probs.values()),
            },
            'fidelidade': {
                'type': 'noul',
                'noul': noul,
            },
        }

    def _avaliar(self, pergunta: str, resposta: str, evidencia: str) -> tuple[str, float]:
        if not resposta or not resposta.strip():
            return 'sem_apoio', 0.0

        resp_lower = resposta.lower().strip()
        perg_lower = pergunta.lower().strip()
        evid_lower = evidencia.lower().strip()

        # 1. Saudações, encerramentos, ofertas de ajuda e abstenções legítimas contam como apoiada
        padroes_apoiados = (
            'olá', 'ola', 'bom dia', 'boa tarde', 'boa noite', 'como posso ajudar',
            'não encontrei', 'nao encontrei', 'não foram encontrados', 'nao foram encontrados',
            'não há registros', 'nao ha registros', 'não consta', 'nao consta',
            'nenhuma cooperativa compatível', 'nenhum registro', 'não tenho evidência',
            'posso ajudar com', 'informações sobre o renovaí',
        )
        if any(p in resp_lower for p in padroes_apoiados) and len(resp_lower) < 250:
            return 'apoiada', 0.95

        # 2. Extração e validação de números significativos na resposta
        numeros_resp_raw = set(re.findall(r'\b\d+(?:[\.,]\d+)*\b', resp_lower))
        numeros_resp_significativos = {
            n for n in numeros_resp_raw
            if len(n) > 1 or n not in {'1', '2', '3', '4', '5'}
        }

        if numeros_resp_significativos:
            numeros_evid_raw = set(re.findall(r'\b\d+(?:[\.,]\d+)*\b', evid_lower + ' ' + perg_lower))
            todas_evid_variantes = set()
            for n in numeros_evid_raw:
                todas_evid_variantes.update(_normalizar_numero(n))

            numeros_faltantes = set()
            for n in numeros_resp_significativos:
                variantes = _normalizar_numero(n)
                if not (variantes & todas_evid_variantes):
                    numeros_faltantes.add(n)

            if numeros_faltantes:
                proporcao_faltante = len(numeros_faltantes) / len(numeros_resp_significativos)
                if proporcao_faltante > 0.4:
                    return 'sem_apoio', 0.12
                else:
                    return 'parcial', 0.45

        # 3. Evidência explicitamente vazia com resposta afirmativa longa
        if ('(nenhuma evidência' in evid_lower or evid_lower == '[]') and len(resp_lower) > 80:
            if not any(neg in resp_lower for neg in ['não', 'nao', 'nenhum', 'nenhuma', 'desculpe', 'infelizmente']):
                return 'sem_apoio', 0.15

        return 'apoiada', 0.88


class JudgeAgent(BaseAgent):
    nome_agente: str = 'judge'
    descricao: str = 'Juiz de fidelidade factual e controle de alucinação.'

    def __init__(
        self,
        llm: Any = None,
        system_prompt: str = '',
        tools: Optional[list] = None,
        limiar_noul: float = LIMIAR_NOUL_PADRAO,
        frase_segura: str = FRASE_SEGURA,
        engine: Optional[DecisionEngine] = None,
    ):
        super().__init__(llm, system_prompt, tools or [])
        self.limiar_noul = limiar_noul
        self.frase_segura = frase_segura

        if engine is not None:
            self._engine = engine
        else:
            try:
                self._engine = LayaDecisionEngine()
                log.info('[judge] Motor LayaDecisionEngine inicializado com sucesso.')
            except Exception as e:
                log.info(f'[judge] Laya indisponível ({e}). Utilizando DeterministicDecisionEngine.')
                self._engine = DeterministicDecisionEngine()

    def _coletar_dados_turno(self, state: GraphState) -> tuple[str, str, str, Any]:
        msgs = list(state.get('messages') or [])
        pergunta = ''
        resposta = ''
        ultima_ai_msg = None

        for m in reversed(msgs):
            tipo = getattr(m, 'type', '')
            role = getattr(m, 'role', '')
            conteudo = self._obter_texto_mensagem(m).strip()
            if not resposta and (tipo == 'ai' or role in ('ai', 'assistant')) and conteudo:
                resposta = conteudo
                ultima_ai_msg = m
            elif not pergunta and (tipo == 'human' or role in ('human', 'user')) and conteudo:
                pergunta = conteudo
            if pergunta and resposta:
                break

        partes_evidencia = []
        snapshots = state.get('snapshots') or {}
        for k, v in snapshots.items():
            partes_evidencia.append(f'[{k}]\n{v}')

        outputs = state.get('especialista_outputs') or {}
        for k, out in outputs.items():
            conteudo = getattr(out, 'conteudo', str(out))
            fonte = getattr(out, 'fonte', k) or k
            partes_evidencia.append(f'[especialista:{fonte}]\n{conteudo}')

        chunks = state.get('chunk_contexto') or []
        for ch in chunks:
            partes_evidencia.append(f'[documento_rag]\n{ch}')

        evidencia = '\n\n'.join(partes_evidencia).strip()
        if not evidencia:
            evidencia = '(nenhuma evidência disponível neste turno)'

        return pergunta, resposta, evidencia[-5000:], ultima_ai_msg

    def _avaliar(self, pergunta: str, resposta: str, evidencia: str) -> tuple[str, float]:
        contexto = {
            'pergunta': pergunta,
            'resposta': resposta,
            'evidencia': evidencia,
            'regras': REGRAS_AVALIACAO,
        }
        questions = [
            {
                'id': 'veredito',
                'type': 'choice',
                'instructions': 'Seguindo as regras, a resposta só afirma o que a evidência sustenta?',
                'criteria': {
                    'apoiada': 'toda afirmação factual aparece na evidência',
                    'parcial': 'parte aparece e parte não',
                    'sem_apoio': 'há fato, número, regra ou opinião sem evidência',
                },
            },
            {
                'id': 'fidelidade',
                'type': 'noul',
                'instructions': (
                    'Seguindo as regras, a resposta é fiel à evidência: todo valor, moeda, '
                    'data e fato dela sai da evidência, da fala do usuário ou de contas sobre elas?'
                ),
            },
        ]
        resultado = self._engine.predict(contexto, questions)
        answers = resultado.get('answers', resultado)

        veredito_data = answers.get('veredito', {})
        choice = veredito_data.get('choice') or 'sem_apoio'

        fidelidade_data = answers.get('fidelidade', {})
        noul = float(fidelidade_data.get('noul', 0.0))

        return choice, noul

    def run(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        pergunta, resposta, evidencia, ultima_ai_msg = self._coletar_dados_turno(state)
        if not resposta:
            return {
                'agentes_chamados': [f'{self.nome_agente}:sem_resposta'],
                'passou_juiz': True,
            }

        try:
            choice, noul = self._avaliar(pergunta, resposta, evidencia)
        except Exception as exc:
            log.error(f'[judge] Falha na inferência do motor de decisão: {exc}. Aplicando fail-safe.')
            choice, noul = 'sem_apoio', 0.0

        aprovado = (choice == 'apoiada' and noul >= self.limiar_noul)
        rotulo = f'apoiada noul={noul:.2f}' if aprovado else f'{choice} noul={noul:.2f}'
        tag_agente = f'{self.nome_agente}:{rotulo}'
        log.info(f'[judge] veredito={tag_agente}')

        if aprovado:
            return {
                'agentes_chamados': [tag_agente],
                'passou_juiz': True,
                'veredito_juiz': 'apoiada',
                'score_juiz': noul,
                'motivo_juiz': rotulo,
            }
        else:
            msg_dict = {'role': 'assistant', 'content': self.frase_segura}
            msg_id = getattr(ultima_ai_msg, 'id', None)
            if msg_id:
                msg_dict['id'] = msg_id
            return {
                'messages': [msg_dict],
                'agentes_chamados': [tag_agente],
                'passou_juiz': False,
                'veredito_juiz': choice,
                'score_juiz': noul,
                'motivo_juiz': rotulo,
                'frase_original': resposta,
            }

    async def arun(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        return self.run(state, config)
