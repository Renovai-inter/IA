import json
import logging
import re
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState

log = logging.getLogger('router')


def _extrair_failed_generation(exc: Exception) -> str | None:
    """Extrai o texto gerado pelo modelo quando a Groq/OpenAI rejeita com 'tool_use_failed'.

    Acontece quando o LLM gera o protocolo de roteamento em formato de tool call
    (ex: `{"name": "ROUTER", "arguments": "ROUTE=..."}`), mas nenhuma tool foi configurada.
    O conteúdo do failed_generation contém o protocolo que o modelo tentou emitir.
    """
    body = getattr(exc, 'body', None)
    if not isinstance(body, dict):
        response = getattr(exc, 'response', None)
        if response and hasattr(response, 'json'):
            try:
                body = response.json()
            except Exception:
                body = None

    if isinstance(body, dict):
        err_dict = body.get('error', {})
        if isinstance(err_dict, dict):
            failed = err_dict.get('failed_generation')
            if failed:
                return str(failed)

    msg = str(exc)
    if 'failed_generation' in msg:
        m = re.search(r"['\"]failed_generation['\"]\s*:\s*(['\"])([\s\S]*?)\1\s*[,}]", msg)
        if m:
            raw = m.group(2)
            try:
                return raw.encode('utf-8').decode('unicode_escape', errors='ignore')
            except Exception:
                return raw
    return None


def _extrair_rotas_protocolo(texto: str) -> list[str]:
    texto_strip = (texto or '').strip()
    if not texto_strip:
        return []

    # 0. Se for um tool_call da Groq/OpenAI (ex: failed_generation: {"name": "ROUTER", "arguments": ...})
    # Remove o envelope do tool call para expor os argumentos diretamente
    texto_limpo = re.sub(
        r'\{?\s*["\']name["\']\s*:\s*["\'][^"\']+["\']\s*,\s*["\']arguments["\']\s*:\s*',
        '',
        texto_strip,
    )

    candidatos_json: list[str] = []
    if '```' in texto_strip:
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", texto_strip, re.IGNORECASE):
            candidatos_json.append(match.group(1).strip())

    match_obj = re.search(r"\{[\s\S]*\}", texto_strip)
    if match_obj:
        candidatos_json.append(match_obj.group(0).strip())

    for bloco in candidatos_json:
        try:
            dados = json.loads(bloco)
            if isinstance(dados, dict):
                # Se vier no formato tool_call {"name": ..., "arguments": ...}
                if 'arguments' in dados:
                    args = dados['arguments']
                    if isinstance(args, dict):
                        for chave in ('ROUTES', 'ROUTE', 'routes', 'route'):
                            if chave in args:
                                val = args[chave]
                                rotas: list[str] = []
                                if isinstance(val, list):
                                    rotas = [str(r).strip().strip("'\"{}") for r in val if str(r).strip()]
                                elif isinstance(val, str):
                                    val_clean = val.strip().strip('[]{}')
                                    rotas = [r.strip().strip("'\"{}") for r in val_clean.split(',') if r.strip().strip("'\"{}")]
                                if rotas:
                                    return [r for r in rotas if r not in ('fim', 'fim_direto')]
                    elif isinstance(args, str):
                        rotas_sub = _extrair_rotas_protocolo(args)
                        if rotas_sub:
                            return rotas_sub

                for chave in ('ROUTES', 'ROUTE', 'routes', 'route'):
                    if chave in dados:
                        val = dados[chave]
                        rotas: list[str] = []
                        if isinstance(val, list):
                            rotas = [str(r).strip().strip("'\"{}") for r in val if str(r).strip()]
                        elif isinstance(val, str):
                            val_clean = val.strip().strip('[]{}')
                            rotas = [r.strip().strip("'\"{}") for r in val_clean.split(',') if r.strip().strip("'\"{}")]
                        if rotas:
                            return [r for r in rotas if r not in ('fim', 'fim_direto')]
        except Exception:
            pass

    for fonte in (texto_limpo, texto):
        rotas_linhas: list[str] = []
        for linha in fonte.splitlines():
            linha_strip = linha.strip()
            linha_limpa = linha_strip.strip("`*\"' {}")
            if linha_limpa.startswith(('ROUTES=', 'ROUTES:')):
                payload = re.split(r'[:=]', linha_limpa, maxsplit=1)[1].strip().strip('[]{}')
                rotas_linhas = [r.strip().strip("'\"{}") for r in payload.split(',') if r.strip().strip("'\"{}")]
                break
            elif linha_limpa.startswith(('ROUTE=', 'ROUTE:')):
                payload = re.split(r'[:=]', linha_limpa, maxsplit=1)[1].strip().strip('[]{}')
                rotas_linhas = [r.strip().strip("'\"{}") for r in payload.split(',') if r.strip().strip("'\"{}")]
                break

        if rotas_linhas:
            return [r for r in rotas_linhas if r not in ('fim', 'fim_direto')]

        match_inline = re.search(
            r"""(?:["'`*]*\bROUTES?\b["'`*]*)\s*[:=]\s*(\[[^\]]*\]|'[^'\n]+'|"[^"\n]+"|[^\s,;\n}]+(?:,\s*[^\s,;\n}]+)*)""",
            fonte,
        )
        if match_inline:
            payload = match_inline.group(1).strip().strip('[]{}')
            rotas = [r.strip().strip("'\"{}") for r in payload.split(',') if r.strip().strip("'\"{}")]
            rotas_filtradas = [r for r in rotas if r not in ('fim', 'fim_direto')]
            if rotas_filtradas:
                return rotas_filtradas

    return []


class RouterAgent(BaseAgent):
    nome_agente: str = 'router'
    descricao: str = 'Agente especialista em roteamento de perguntas para especialistas.'

    def __init__(self, llm, system_prompt, tools = None):
        super().__init__(llm, system_prompt, tools)

        self._runnable = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
        )

    # O roteador só decide a rota: precisa da pergunta atual e de um pouco de contexto
    # para follow-ups ("e o de papel?"). As respostas anteriores dos especialistas (JSON
    # longo) inflavam a entrada (~2,5 mil tokens/chamada) e estouravam o TPM do Groq.
    janela_mensagens: int = 4
    _MAX_CHARS_CONTEXTO: int = 400

    def _mensagens_recentes(self, state: GraphState) -> list:
        recentes = super()._mensagens_recentes(state)
        if len(recentes) <= 1:
            return recentes
        *anteriores, ultima = recentes
        return [*(self._truncar_mensagem(m) for m in anteriores), ultima]

    def _truncar_mensagem(self, msg):
        texto = self._obter_texto_mensagem(msg)
        if len(texto) <= self._MAX_CHARS_CONTEXTO:
            return msg
        cortado = texto[:self._MAX_CHARS_CONTEXTO] + '…'
        if isinstance(msg, dict):
            return {**msg, 'content': cortado}
        try:
            return msg.model_copy(update={'content': cortado})
        except Exception:
            return msg

    def _montar_mensagens_llm(self, state: GraphState) -> list:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
        mensagens: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        for m in self._mensagens_recentes(state):
            if isinstance(m, BaseMessage):
                mensagens.append(m)
            elif isinstance(m, dict):
                role = str(m.get('role') or m.get('type') or '')
                texto = self._obter_texto_mensagem(m)
                if role in ('human', 'user'):
                    mensagens.append(HumanMessage(content=texto))
                elif role in ('ai', 'assistant'):
                    mensagens.append(AIMessage(content=texto))
                elif role == 'system':
                    mensagens.append(SystemMessage(content=texto))
                else:
                    mensagens.append(HumanMessage(content=texto))
        return mensagens

    def _processar_resultado(self, resultado: dict) -> dict:
        texto_resposta = self._obter_texto_mensagem(resultado['messages'][-1])

        rotas = _extrair_rotas_protocolo(texto_resposta)
        eh_fim = len(rotas) == 0

        # O protocolo ROUTES=.../PERGUNTA_ORIGINAL=... é comunicação interna
        # roteador -> grafo: só vira mensagem no histórico quando o roteador
        # está respondendo direto ao usuário (saudação / fora de escopo).
        # Quando encaminha, os especialistas leem a pergunta do próprio
        # HumanMessage, e as rotas seguem em `proximos_agentes`.
        mensagens = (
            [{'role': 'assistant', 'content': texto_resposta}]
            if eh_fim else []
        )

        return {
            'messages'        : mensagens,
            'proximos_agentes': rotas,
            'agentes_chamados': [self.nome_agente],
        }

    def run(self, state: GraphState, config: RunnableConfig) -> dict:
        if self.llm is not None and not self.tools:
            msgs = self._montar_mensagens_llm(state)
            try:
                resposta = self.llm.invoke(msgs, config=self._config_interno(config))
                return self._processar_resultado({'messages': [resposta]})
            except Exception as exc:
                failed = _extrair_failed_generation(exc)
                if failed:
                    log.warning('router_agent.run: Groq tool_use_failed capturado; extraindo rotas de failed_generation')
                    return self._processar_resultado({'messages': [{'role': 'assistant', 'content': failed}]})
                raise

        try:
            resultado = self._runnable.invoke(
                {'messages': self._mensagens_recentes(state)},
                config=self._config_interno(config)
            )
            return self._processar_resultado(resultado)
        except Exception as exc:
            failed = _extrair_failed_generation(exc)
            if failed:
                log.warning('router_agent.run: Groq tool_use_failed capturado em _runnable; extraindo rotas de failed_generation')
                return self._processar_resultado({'messages': [{'role': 'assistant', 'content': failed}]})
            raise

    async def arun(self, state: GraphState, config: RunnableConfig) -> dict:
        if self.llm is not None and not self.tools:
            msgs = self._montar_mensagens_llm(state)
            try:
                resposta = await self.llm.ainvoke(msgs, config=self._config_interno(config))
                return self._processar_resultado({'messages': [resposta]})
            except Exception as exc:
                failed = _extrair_failed_generation(exc)
                if failed:
                    log.warning('router_agent.arun: Groq tool_use_failed capturado; extraindo rotas de failed_generation')
                    return self._processar_resultado({'messages': [{'role': 'assistant', 'content': failed}]})
                raise

        try:
            resultado = await self._runnable.ainvoke(
                {'messages': self._mensagens_recentes(state)},
                config=self._config_interno(config)
            )
            return self._processar_resultado(resultado)
        except Exception as exc:
            failed = _extrair_failed_generation(exc)
            if failed:
                log.warning('router_agent.arun: Groq tool_use_failed capturado em _runnable; extraindo rotas de failed_generation')
                return self._processar_resultado({'messages': [{'role': 'assistant', 'content': failed}]})
            raise
