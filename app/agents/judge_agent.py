"""Agente Juiz (JudgeAgent) — Controle de Alucinação e Verificação de Evidências via Laya Studio API.

Implementa um modelo de decisão discriminativo ('System One') no portal terminal do grafo,
consumindo a API oficial do Laya Studio (`POST https://api.laya.studio/v1/systemone`) via `LAYA_API_KEY`.
Avalia a resposta final (orquestrada/sintetizada) contra as evidências factuais do turno
(snapshots de banco, saídas de especialistas e chunks RAG do Qdrant/PNRS).

Não redige texto livre, não carrega pesos pesados em RAM local e não reinjeta ciclos lentos de reflexão.
Se a resposta for fiel (`choice == 'apoiada'` e `noul >= limiar`), o texto original segue intacto.
Se a resposta for infundada, extrapolar a evidência, ou ocorrer falha/ausência de chave, ela é
substituída de forma determinística pela frase segura pré-definida (`FRASE_SEGURA`).
"""
from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Optional

import httpx
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState

log = logging.getLogger('renovai.judge')

DEFAULT_LAYA_API_URL = 'https://api.laya.studio/v1/systemone'
DEFAULT_LAYA_MODEL = 'multilingual'

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
    'Números inventados, moedas trocadas, opiniões tratadas como fato do sistema, '
    'valores sem evidência ou afirmações factuais não suportadas contam como sem_apoio.'
)


class LayaMissingKeyError(RuntimeError):
    """Exceção disparada quando LAYA_API_KEY não está configurada."""


class DecisionEngine:
    """Contrato base para o motor de decisão System One do Juiz."""

    def predict(self, state: dict, questions: list[dict] | dict) -> dict:
        raise NotImplementedError

    async def apredict(self, state: dict, questions: list[dict] | dict) -> dict:
        return self.predict(state, questions)


class LayaDecisionEngine(DecisionEngine):
    """Cliente HTTP para o motor de decisão System One do Laya Studio (`laya.studio`).

    Compatível com o protocolo `/v1/systemone` (JEV / Laya Studio), enviando `model`,
    `state` e `questions` (`choice` e `noul`) e recebendo probabilidades calibradas
    sem carregar PyTorch ou pesos do Hugging Face em memória RAM local.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.api_key = (api_key if api_key is not None else os.getenv('LAYA_API_KEY', '')).strip()
        self.api_url = (api_url or os.getenv('LAYA_API_URL') or DEFAULT_LAYA_API_URL).strip()
        self.model = (model or os.getenv('LAYA_MODEL') or DEFAULT_LAYA_MODEL).strip()
        self.timeout = timeout

    @staticmethod
    def _normalizar_questions(questions: list[dict] | dict) -> dict[str, dict]:
        if isinstance(questions, list):
            return {
                q.get('id', str(i)): {k: v for k, v in q.items() if k != 'id'}
                for i, q in enumerate(questions)
            }
        return dict(questions)

    def _montar_request(self, state: dict, questions: list[dict] | dict) -> tuple[dict[str, str], dict[str, Any]]:
        if not self.api_key:
            raise LayaMissingKeyError('LAYA_API_KEY não configurada.')

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload: dict[str, Any] = {
            'model': self.model,
            'state': state,
            'questions': self._normalizar_questions(questions),
        }
        return headers, payload

    def predict(self, state: dict, questions: list[dict] | dict) -> dict:
        headers, payload = self._montar_request(state, questions)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def apredict(self, state: dict, questions: list[dict] | dict) -> dict:
        headers, payload = self._montar_request(state, questions)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()


class JudgeAgent(BaseAgent):
    nome_agente: str = 'judge'
    descricao: str = 'Juiz de fidelidade factual e controle de alucinação via Laya Studio API.'

    def __init__(
        self,
        llm: Any = None,
        system_prompt: str = '',
        tools: Optional[list] = None,
        limiar_noul: float = LIMIAR_NOUL_PADRAO,
        frase_segura: str = FRASE_SEGURA,
        engine: Optional[DecisionEngine] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 15.0,
    ):
        super().__init__(llm, system_prompt, tools or [])
        self.limiar_noul = limiar_noul
        self.frase_segura = frase_segura
        self._engine: DecisionEngine = engine or LayaDecisionEngine(
            api_key=api_key,
            api_url=api_url,
            model=model,
            timeout=timeout,
        )

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

        return pergunta, resposta, evidencia[-6000:], ultima_ai_msg

    @staticmethod
    def _construir_payload_avaliacao(pergunta: str, resposta: str, evidencia: str) -> tuple[dict, dict]:
        contexto = {
            'pergunta': pergunta,
            'resposta': resposta,
            'evidencia': evidencia,
            'regras': REGRAS_AVALIACAO,
        }
        questions = {
            'veredito': {
                'type': 'choice',
                'instructions': 'Seguindo as regras, a resposta só afirma o que a evidência sustenta?',
                'criteria': {
                    'apoiada': 'toda afirmação factual aparece na evidência',
                    'parcial': 'parte aparece e parte não',
                    'sem_apoio': 'há fato, número, regra ou opinião sem evidência',
                },
            },
            'fidelidade': {
                'type': 'noul',
                'instructions': (
                    'Seguindo as regras, a resposta é fiel à evidência: todo valor, moeda, '
                    'data e fato dela sai da evidência, da fala do usuário ou de contas sobre elas?'
                ),
            },
        }
        return contexto, questions

    @staticmethod
    def _extrair_veredito(resultado: dict) -> tuple[str, float]:
        usage = resultado.get('usage')
        if isinstance(usage, dict):
            from app.core import observabilidade as obs_mod
            inst = getattr(obs_mod, '_instancia_observabilidade', None)
            if inst is not None:
                modelo_resp = str(resultado.get('model') or 'laya-multilingual')
                inst.registrar_chamada_externa(
                    modelo=modelo_resp,
                    no='judge',
                    tokens_entrada=int(usage.get('input_tokens') or 0),
                    tokens_saida=int(usage.get('output_tokens') or 0),
                )

        answers = resultado.get('answers', resultado)
        veredito_data = answers.get('veredito', {})
        choice = veredito_data.get('choice') or 'sem_apoio'

        fidelidade_data = answers.get('fidelidade') or answers.get('apoiada') or {}
        noul = float(fidelidade_data.get('noul', 0.0))
        return choice, noul

    def _avaliar(self, pergunta: str, resposta: str, evidencia: str) -> tuple[str, float]:
        contexto, questions = self._construir_payload_avaliacao(pergunta, resposta, evidencia)
        resultado = self._engine.predict(contexto, questions)
        return self._extrair_veredito(resultado)

    async def _avaliar_async(self, pergunta: str, resposta: str, evidencia: str) -> tuple[str, float]:
        contexto, questions = self._construir_payload_avaliacao(pergunta, resposta, evidencia)
        apredict_fn = getattr(self._engine, 'apredict', None)
        if apredict_fn is not None and inspect.iscoroutinefunction(apredict_fn):
            resultado = await apredict_fn(contexto, questions)
        else:
            resultado = self._engine.predict(contexto, questions)
        return self._extrair_veredito(resultado)

    def _formatar_saida(
        self,
        choice: str,
        noul: float,
        resposta: str,
        ultima_ai_msg: Any,
        rotulo_erro: Optional[str] = None,
    ) -> dict:
        if rotulo_erro:
            aprovado = False
            rotulo = rotulo_erro
        else:
            aprovado = choice == 'apoiada' and noul >= self.limiar_noul
            if aprovado:
                rotulo = f'apoiada noul={noul:.2f}'
            elif choice == 'apoiada':
                rotulo = f'apoiada:abaixo_do_limiar noul={noul:.2f}'
            else:
                rotulo = f'{choice} noul={noul:.2f}'

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

        msg_dict: dict[str, Any] = {'role': 'assistant', 'content': self.frase_segura}
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

    def run(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        pergunta, resposta, evidencia, ultima_ai_msg = self._coletar_dados_turno(state)
        if not resposta:
            return {
                'agentes_chamados': [f'{self.nome_agente}:sem_resposta'],
                'passou_juiz': True,
            }

        rotulo_erro: Optional[str] = None
        try:
            choice, noul = self._avaliar(pergunta, resposta, evidencia)
        except LayaMissingKeyError:
            log.warning('[judge] LAYA_API_KEY ausente. Aplicando fail-safe (sem_chave).')
            choice, noul, rotulo_erro = 'sem_chave', 0.0, 'sem_chave'
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            log.error(f'[judge] Erro HTTP {code} ao consultar Laya Studio API. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro_http', 0.0, f'erro_http_{code}'
        except Exception as exc:
            log.error(f'[judge] Falha na inferência do Laya Studio API: {exc}. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro', 0.0, 'erro'

        return self._formatar_saida(choice, noul, resposta, ultima_ai_msg, rotulo_erro=rotulo_erro)

    async def arun(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        pergunta, resposta, evidencia, ultima_ai_msg = self._coletar_dados_turno(state)
        if not resposta:
            return {
                'agentes_chamados': [f'{self.nome_agente}:sem_resposta'],
                'passou_juiz': True,
            }

        rotulo_erro: Optional[str] = None
        try:
            choice, noul = await self._avaliar_async(pergunta, resposta, evidencia)
        except LayaMissingKeyError:
            log.warning('[judge] LAYA_API_KEY ausente. Aplicando fail-safe (sem_chave).')
            choice, noul, rotulo_erro = 'sem_chave', 0.0, 'sem_chave'
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            log.error(f'[judge] Erro HTTP {code} ao consultar Laya Studio API. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro_http', 0.0, f'erro_http_{code}'
        except Exception as exc:
            log.error(f'[judge] Falha na inferência do Laya Studio API: {exc}. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro', 0.0, 'erro'

        return self._formatar_saida(choice, noul, resposta, ultima_ai_msg, rotulo_erro=rotulo_erro)
