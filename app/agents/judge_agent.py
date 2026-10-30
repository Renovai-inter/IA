"""Agente Juiz (JudgeAgent) — Controle de Alucinação e Verificação de Evidências via Gemini (System One).

Implementa um modelo de decisão discriminativo ('System One') no portal terminal do grafo,
utilizando o Google Gemini com temperatura 0.0, saída estritamente estruturada em JSON e
thinking de baixa latência para validar a resposta final contra as evidências factuais do turno
(snapshots de banco, saídas de especialistas e chunks RAG do Qdrant/PNRS).

Não redige texto livre, não reinjeta ciclos lentos de reflexão e finaliza a decisão em frações de segundo.
Se a resposta for fiel (`choice == 'apoiada'` e `noul >= limiar`), o texto original segue intacto.
Se a resposta for infundada ou extrapolar a evidência, ela é substituída de forma determinística
pela frase segura pré-definida (`FRASE_SEGURA`).

Se o próprio Juiz não conseguir avaliar (chave ausente, erro HTTP, cota, timeout), a resposta segue
com o rótulo `judge:indisponivel:<motivo>`: ela já passou pelos guardrails, e trocar toda resposta
pela frase segura deixaria o chat inteiro sem uso enquanto o motor do Juiz estiver fora (foi o que
aconteceu com o 402 do Laya). Com `JUIZ_BLOQUEAR_SEM_VEREDITO=true` volta o bloqueio também nesse caso.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
from typing import Any, Optional

import httpx
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState

log = logging.getLogger('renovai.judge')

DEFAULT_GEMINI_JUDGE_MODEL = 'gemini-3.5-flash-lite'
FALLBACK_GEMINI_JUDGE_MODEL = 'gemini-2.5-flash'

FRASE_SEGURA = (
    'Não localizei evidências suficientes nos registros operacionais da cooperativa '
    'ou nas normas vigentes para confirmar essas informações com exatidão.'
)
LIMIAR_NOUL_PADRAO = 0.70


def _bloquear_sem_veredito_env() -> bool:
    return (os.getenv('JUIZ_BLOQUEAR_SEM_VEREDITO') or '').strip().lower() in ('1', 'true', 'sim', 'yes')

REGRAS_AVALIACAO = (
    'A pergunta do usuário é evidência do que ele solicitou, declarou ou informou ter feito. '
    'Somas, contagens, conclusões lógicas, correlações analíticas entre múltiplos especialistas, '
    'cálculos de proporção/diferença, apontamento de divergências entre registros, sugestões de '
    'próximo passo, ofertas de ajuda, saudações e confirmação de ausência de registros contam como apoiada. '
    'Valores numéricos de estoque (kg), preços médios, custos, despesas, datas de coleta, '
    'triagens, rotas de veículos, nomes de cooperativas e artigos da lei PNRS devem '
    'estar estritamente fundamentados na evidência disponível. '
    'Números inventados, moedas trocadas, opiniões tratadas como fato do sistema, '
    'valores sem evidência ou afirmações factuais não suportadas contam como sem_apoio.'
)


_PADROES_SAUDACAO_OU_CORTESIA = re.compile(
    r'\b(ol[áa]|bom dia|boa tarde|boa noite|como posso ajudar|em que posso ajudar|'
    r'como posso te ajudar|estou [àa] disposi[çc][ãa]o|dispor|at[ée] logo|at[ée] mais|'
    r'tchau|obrigad[oa]|por nada|de nada|por favor[, ]+(?:informe|especifique|poderia)|'
    r'posso te ajudar com)\b',
    re.IGNORECASE,
)
_PADROES_DADOS_FACTUAIS = re.compile(
    r'(\d+[\.,]?\d*|\bR\$\b|\bUS\$\b|\bkg\b|\bton\b|\btoneladas?\b|\blitros?\b|\bkm\b|\bcep\b|\bcpf\b|\bcnpj\b|\blei\b|\bartigo\b)',
    re.IGNORECASE,
)


class GeminiMissingKeyError(RuntimeError):
    """Exceção disparada quando GEMINI_API_KEY não está configurada."""


class DecisionEngine:
    """Contrato base para o motor de decisão System One do Juiz."""

    def predict(self, state: dict, questions: list[dict] | dict) -> dict:
        raise NotImplementedError

    async def apredict(self, state: dict, questions: list[dict] | dict) -> dict:
        return self.predict(state, questions)


class GeminiDecisionEngine(DecisionEngine):
    """Motor de decisão System One principal baseado no Gemini (temperatura 0.0).

    Atua exclusivamente como classificador discriminativo estruturado (sem geração de texto livre),
    respeitando o protocolo System One e retornando probabilidades calibradas de 'choice' e 'noul'.
    Otimizado para baixa latência (thinking='low', max_retries=1, timeout controlado).
    """

    def __init__(
        self,
        llm: Any = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 1,
    ):
        self.api_key = (api_key or os.getenv('GEMINI_API_KEY') or '').strip()
        self.model = (model or DEFAULT_GEMINI_JUDGE_MODEL).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self._llm = llm

    def _obter_llm(self, model_name: Optional[str] = None):
        target_model = model_name or self.model
        if self._llm is not None:
            if model_name is None or getattr(self._llm, 'model', None) == target_model:
                return self._llm

        if not self.api_key:
            raise GeminiMissingKeyError('GEMINI_API_KEY não configurada para o GeminiDecisionEngine.')
        from langchain_google_genai import ChatGoogleGenerativeAI
        from app.llms.tuning import kwargs_suportados

        kwargs = kwargs_suportados(
            ChatGoogleGenerativeAI,
            thinking_level='low',
            max_retries=self.max_retries,
            timeout=self.timeout,
            max_output_tokens=256,
            response_mime_type='application/json',
        )
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=self.api_key,
            temperature=0.0,
            **kwargs,
        )

    def _montar_prompt(self, state: dict) -> str:
        pergunta = state.get('pergunta', '')
        resposta = state.get('resposta', '')
        evidencia = state.get('evidencia', '')
        regras = state.get('regras', REGRAS_AVALIACAO)
        return (
            'Você é um classificador discriminativo estrito de fidelidade factual (System One).\n'
            'Sua única função é avaliar se a resposta afirmada pelo sistema está estritamente '
            'fundamentada na evidência disponível.\n'
            'NÃO gere explicações, NÃO converse, NÃO justifique e NÃO reescreva o texto.\n\n'
            f'REGRAS DE AVALIAÇÃO:\n{regras}\n\n'
            f'PERGUNTA DO USUÁRIO:\n{pergunta}\n\n'
            f'EVIDÊNCIAS DISPONÍVEIS:\n{evidencia}\n\n'
            f'RESPOSTA GERADA:\n{resposta}\n\n'
            'Responda EXCLUSIVAMENTE um objeto JSON válido com este formato:\n'
            '{\n'
            '  "choice": "apoiada",\n'
            '  "noul": 0.95\n'
            '}\n'
            'Onde:\n'
            '- "choice" deve ser exatamente "apoiada" ou "sem_apoio".\n'
            '- "noul" deve ser um float entre 0.00 e 1.00 representando a probabilidade de fidelidade.'
        )

    @staticmethod
    def _parsear_resposta(
        texto: str,
        usage_metadata: Optional[dict] = None,
        model_name: str = DEFAULT_GEMINI_JUDGE_MODEL,
    ) -> dict:
        texto_limpo = texto.strip()
        match = re.search(r'\{.*\}', texto_limpo, re.DOTALL)
        if match:
            texto_limpo = match.group(0)
        try:
            dados = json.loads(texto_limpo)
            choice_raw = str(dados.get('choice', 'sem_apoio')).lower().strip()
            choice = 'apoiada' if 'apoiada' in choice_raw else 'sem_apoio'
            noul = float(dados.get('noul', 0.0))
            noul = max(0.0, min(1.0, noul))
        except Exception:
            choice = 'sem_apoio'
            noul = 0.0

        res: dict[str, Any] = {
            'model': model_name,
            'answers': {
                'veredito': {'type': 'choice', 'choice': choice},
                'fidelidade': {'type': 'noul', 'noul': noul},
            },
            '_origem': 'gemini',
        }
        if usage_metadata and isinstance(usage_metadata, dict):
            res['usage'] = {
                'input_tokens': usage_metadata.get('input_tokens', 0),
                'output_tokens': usage_metadata.get('output_tokens', 0),
            }
        return res

    def predict(self, state: dict, questions: list[dict] | dict) -> dict:
        prompt = self._montar_prompt(state)
        current_model = getattr(self._llm, 'model', getattr(self._llm, 'model_name', self.model))
        try:
            llm = self._obter_llm(current_model)
            resp = llm.invoke(prompt)
        except Exception as exc:
            err_str = str(exc)
            if any(k in err_str for k in ('404', 'NOT_FOUND', '504', 'DEADLINE_EXCEEDED', 'Deadline', '429', 'RESOURCE_EXHAUSTED', 'ResourceExhausted', 'Quota', 'Too Many Requests', 'rate limit')):
                fallback_mod = FALLBACK_GEMINI_JUDGE_MODEL
                if current_model != fallback_mod:
                    log.warning(
                        f'[judge] Falha com modelo principal {current_model} ({exc}). '
                        f'Tentando modelo de contingência {fallback_mod}...'
                    )
                    current_model = fallback_mod
                    llm = self._obter_llm(current_model)
                    resp = llm.invoke(prompt)
                else:
                    raise
            else:
                raise
        conteudo = getattr(resp, 'content', '')
        if isinstance(conteudo, list):
            conteudo = ''.join(b.get('text', '') for b in conteudo if isinstance(b, dict))
        usage = getattr(resp, 'usage_metadata', None)
        return self._parsear_resposta(str(conteudo), usage_metadata=usage, model_name=current_model)

    async def apredict(self, state: dict, questions: list[dict] | dict) -> dict:
        prompt = self._montar_prompt(state)
        current_model = getattr(self._llm, 'model', getattr(self._llm, 'model_name', self.model))
        try:
            llm = self._obter_llm(current_model)
            resp = await llm.ainvoke(prompt)
        except Exception as exc:
            err_str = str(exc)
            if any(k in err_str for k in ('404', 'NOT_FOUND', '504', 'DEADLINE_EXCEEDED', 'Deadline', '429', 'RESOURCE_EXHAUSTED', 'ResourceExhausted', 'Quota', 'Too Many Requests', 'rate limit')):
                fallback_mod = FALLBACK_GEMINI_JUDGE_MODEL
                if current_model != fallback_mod:
                    log.warning(
                        f'[judge] Falha com modelo principal {current_model} ({exc}). '
                        f'Tentando modelo de contingência {fallback_mod}...'
                    )
                    current_model = fallback_mod
                    llm = self._obter_llm(current_model)
                    resp = await llm.ainvoke(prompt)
                else:
                    raise
            else:
                raise
        conteudo = getattr(resp, 'content', '')
        if isinstance(conteudo, list):
            conteudo = ''.join(b.get('text', '') for b in conteudo if isinstance(b, dict))
        usage = getattr(resp, 'usage_metadata', None)
        return self._parsear_resposta(str(conteudo), usage_metadata=usage, model_name=current_model)


class JudgeAgent(BaseAgent):
    nome_agente: str = 'judge'
    descricao: str = 'Juiz de fidelidade factual e controle de alucinação via Gemini (System One).'

    def __init__(
        self,
        llm: Any = None,
        system_prompt: str = '',
        tools: Optional[list] = None,
        limiar_noul: float = LIMIAR_NOUL_PADRAO,
        frase_segura: str = FRASE_SEGURA,
        engine: Optional[DecisionEngine] = None,
        fallback_engine: Optional[DecisionEngine] = None,
        gemini_api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 1,
        fast_path: bool = True,
        bloquear_sem_veredito: Optional[bool] = None,
    ):
        super().__init__(llm, system_prompt, tools or [])
        self.limiar_noul = limiar_noul
        self.frase_segura = frase_segura
        self.fast_path = fast_path
        # sem veredito (Juiz fora do ar): bloqueia a resposta (True) ou deixa seguir rotulada (False)
        self.bloquear_sem_veredito = (
            _bloquear_sem_veredito_env() if bloquear_sem_veredito is None else bloquear_sem_veredito
        )

        if engine is not None:
            self._engine: DecisionEngine = engine
        else:
            gem_key = (gemini_api_key or os.getenv('GEMINI_API_KEY') or '').strip()
            self._engine = GeminiDecisionEngine(
                llm=llm,
                api_key=gem_key,
                model=model or DEFAULT_GEMINI_JUDGE_MODEL,
                timeout=timeout,
                max_retries=max_retries,
            )
        self._fallback_engine: Optional[DecisionEngine] = fallback_engine

    def _coletar_dados_turno(self, state: GraphState) -> tuple[str, str, str, Any]:
        msgs = list(state.get('messages') or [])
        pergunta = ''
        resposta = ''
        ultima_ai_msg = None

        for m in reversed(msgs):
            if isinstance(m, dict):
                tipo = m.get('type') or m.get('role') or ''
                role = m.get('role') or ''
            else:
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
                modelo_resp = str(resultado.get('model') or DEFAULT_GEMINI_JUDGE_MODEL)
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

    def _avaliar(self, pergunta: str, resposta: str, evidencia: str) -> tuple[str, float, str]:
        contexto, questions = self._construir_payload_avaliacao(pergunta, resposta, evidencia)
        origem = 'gemini'
        try:
            resultado = self._engine.predict(contexto, questions)
            origem = resultado.get('_origem', 'gemini')
        except Exception as exc:
            if self._fallback_engine is not None:
                log.warning(
                    f'[judge] Falha no motor principal do Juiz ({exc}). '
                    'Ativando fallback de contingência...'
                )
                resultado = self._fallback_engine.predict(contexto, questions)
                origem = resultado.get('_origem', 'fallback')
            else:
                raise
        choice, noul = self._extrair_veredito(resultado)
        return choice, noul, origem

    async def _avaliar_async(self, pergunta: str, resposta: str, evidencia: str) -> tuple[str, float, str]:
        contexto, questions = self._construir_payload_avaliacao(pergunta, resposta, evidencia)
        origem = 'gemini'
        try:
            apredict_fn = getattr(self._engine, 'apredict', None)
            if apredict_fn is not None and inspect.iscoroutinefunction(apredict_fn):
                resultado = await apredict_fn(contexto, questions)
            else:
                resultado = self._engine.predict(contexto, questions)
            origem = resultado.get('_origem', 'gemini')
        except Exception as exc:
            if self._fallback_engine is not None:
                log.warning(
                    f'[judge] Falha no motor principal do Juiz ({exc}). '
                    'Ativando fallback de contingência...'
                )
                apredict_fb = getattr(self._fallback_engine, 'apredict', None)
                if apredict_fb is not None and inspect.iscoroutinefunction(apredict_fb):
                    resultado = await apredict_fb(contexto, questions)
                else:
                    resultado = self._fallback_engine.predict(contexto, questions)
                origem = resultado.get('_origem', 'fallback')
            else:
                raise
        choice, noul = self._extrair_veredito(resultado)
        return choice, noul, origem

    def _formatar_saida(
        self,
        choice: str,
        noul: float,
        resposta: str,
        ultima_ai_msg: Any,
        rotulo_erro: Optional[str] = None,
        origem: str = 'gemini',
    ) -> dict:
        sufixo_origem = f' ({origem})' if origem != 'gemini' else ''
        if rotulo_erro and not self.bloquear_sem_veredito:
            tag_agente = f'{self.nome_agente}:indisponivel:{rotulo_erro}'
            log.warning(f'[judge] Sem veredito ({rotulo_erro}): resposta liberada sem verificação.')
            return {
                'agentes_chamados': [tag_agente],
                'passou_juiz': True,
                'veredito_juiz': 'indisponivel',
                'score_juiz': None,
                'motivo_juiz': rotulo_erro,
            }
        if rotulo_erro:
            aprovado = False
            rotulo = rotulo_erro
        else:
            aprovado = choice == 'apoiada' and noul >= self.limiar_noul
            if aprovado:
                rotulo = f'apoiada noul={noul:.2f}{sufixo_origem}'
            elif choice == 'apoiada':
                rotulo = f'apoiada:abaixo_do_limiar noul={noul:.2f}{sufixo_origem}'
            else:
                rotulo = f'{choice} noul={noul:.2f}{sufixo_origem}'

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
        msg_id = getattr(ultima_ai_msg, 'id', None) if not isinstance(ultima_ai_msg, dict) else ultima_ai_msg.get('id')
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

    def _eh_fast_path(self, pergunta: str, resposta: str, state: GraphState) -> bool:
        """Determina se a resposta do turno é uma saudação, esclarecimento ou resposta curta
        sem nenhuma alegação factual que demande inferência no motor System One do Juiz."""
        if not resposta:
            return True

        # Se houver números, moedas, pesos, distâncias ou citações de lei, NÃO é fast path
        if _PADROES_DADOS_FACTUAIS.search(resposta):
            return False

        # Se a resposta contiver padrão de saudação/cortesia/ajuda e tiver tamanho razoável (< 300 chars)
        if len(resposta) < 300 and _PADROES_SAUDACAO_OU_CORTESIA.search(resposta):
            return True

        # Se nenhum especialista foi chamado nem há snapshots (resposta direta do roteador) e sem fatos
        tem_especialista = bool(state.get('especialista_outputs'))
        tem_snapshot = bool(state.get('snapshots'))
        tem_rag = bool(state.get('chunk_contexto'))
        if not tem_especialista and not tem_snapshot and not tem_rag and len(resposta) < 250:
            return True

        return False

    def run(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        pergunta, resposta, evidencia, ultima_ai_msg = self._coletar_dados_turno(state)
        if not resposta:
            return {
                'agentes_chamados': [f'{self.nome_agente}:sem_resposta'],
                'passou_juiz': True,
            }

        if self.fast_path and self._eh_fast_path(pergunta, resposta, state):
            log.info(f'[judge] Fast-Path ativado para resposta curta/saudação ({resposta[:50]!r}). Aprovada em 0ms.')
            return self._formatar_saida(
                choice='apoiada',
                noul=1.0,
                resposta=resposta,
                ultima_ai_msg=ultima_ai_msg,
                origem='fast_path',
            )

        rotulo_erro: Optional[str] = None
        origem = 'gemini'
        try:
            choice, noul, origem = self._avaliar(pergunta, resposta, evidencia)
        except GeminiMissingKeyError:
            log.warning('[judge] Chave de API ausente. Aplicando fail-safe (sem_chave).')
            choice, noul, rotulo_erro = 'sem_chave', 0.0, 'sem_chave'
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            log.error(f'[judge] Erro HTTP {code} ao consultar motor do Juiz. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro_http', 0.0, f'erro_http_{code}'
        except Exception as exc:
            log.error(f'[judge] Falha na inferência do Juiz: {exc}. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro', 0.0, 'erro'

        return self._formatar_saida(
            choice, noul, resposta, ultima_ai_msg, rotulo_erro=rotulo_erro, origem=origem
        )

    async def arun(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        pergunta, resposta, evidencia, ultima_ai_msg = self._coletar_dados_turno(state)
        if not resposta:
            return {
                'agentes_chamados': [f'{self.nome_agente}:sem_resposta'],
                'passou_juiz': True,
            }

        if self.fast_path and self._eh_fast_path(pergunta, resposta, state):
            log.info(f'[judge] Fast-Path ativado para resposta curta/saudação ({resposta[:50]!r}). Aprovada em 0ms.')
            return self._formatar_saida(
                choice='apoiada',
                noul=1.0,
                resposta=resposta,
                ultima_ai_msg=ultima_ai_msg,
                origem='fast_path',
            )

        rotulo_erro: Optional[str] = None
        origem = 'gemini'
        try:
            choice, noul, origem = await self._avaliar_async(pergunta, resposta, evidencia)
        except GeminiMissingKeyError:
            log.warning('[judge] Chave de API ausente. Aplicando fail-safe (sem_chave).')
            choice, noul, rotulo_erro = 'sem_chave', 0.0, 'sem_chave'
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            log.error(f'[judge] Erro HTTP {code} ao consultar motor do Juiz. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro_http', 0.0, f'erro_http_{code}'
        except Exception as exc:
            log.error(f'[judge] Falha na inferência do Juiz: {exc}. Aplicando fail-safe.')
            choice, noul, rotulo_erro = 'erro', 0.0, 'erro'

        return self._formatar_saida(
            choice, noul, resposta, ultima_ai_msg, rotulo_erro=rotulo_erro, origem=origem
        )
