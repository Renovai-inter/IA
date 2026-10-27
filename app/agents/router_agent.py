import json
import re
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import GraphState


def _extrair_rotas_protocolo(texto: str) -> list[str]:
    texto_strip = (texto or "").strip()
    if not texto_strip:
        return []

    candidatos_json: list[str] = []
    if "```" in texto_strip:
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", texto_strip, re.IGNORECASE):
            candidatos_json.append(match.group(1).strip())

    match_obj = re.search(r"\{[\s\S]*\}", texto_strip)
    if match_obj:
        candidatos_json.append(match_obj.group(0).strip())

    for bloco in candidatos_json:
        try:
            dados = json.loads(bloco)
            if isinstance(dados, dict):
                for chave in ("ROUTES", "ROUTE", "routes", "route"):
                    if chave in dados:
                        val = dados[chave]
                        rotas: list[str] = []
                        if isinstance(val, list):
                            rotas = [str(r).strip().strip("'\"") for r in val if str(r).strip()]
                        elif isinstance(val, str):
                            val_clean = val.strip().strip("[]")
                            rotas = [r.strip().strip("'\"") for r in val_clean.split(",") if r.strip().strip("'\"")]
                        if rotas:
                            return [r for r in rotas if r not in ('fim', 'fim_direto')]
        except Exception:
            pass

    rotas_linhas: list[str] = []
    for linha in texto.splitlines():
        linha_strip = linha.strip()
        linha_limpa = linha_strip.strip("`*\"' ")
        if linha_limpa.startswith(('ROUTES=', 'ROUTES:')):
            payload = re.split(r'[:=]', linha_limpa, maxsplit=1)[1].strip().strip('[]')
            rotas_linhas = [r.strip().strip("'\"") for r in payload.split(',') if r.strip().strip("'\"")]
            break
        elif linha_limpa.startswith(('ROUTE=', 'ROUTE:')):
            payload = re.split(r'[:=]', linha_limpa, maxsplit=1)[1].strip().strip('[]')
            rotas_linhas = [r.strip().strip("'\"") for r in payload.split(',') if r.strip().strip("'\"")]
            break

    if rotas_linhas:
        return [r for r in rotas_linhas if r not in ('fim', 'fim_direto')]

    match_inline = re.search(
        r"""(?:["'`*]*\bROUTES?\b["'`*]*)\s*[:=]\s*(\[[^\]]*\]|"[^"\n]+"|'[^'\n]+'|[^\s,;\n]+(?:,\s*[^\s,;\n]+)*)""",
        texto,
    )
    if match_inline:
        payload = match_inline.group(1).strip().strip('[]')
        rotas = [r.strip().strip("'\"") for r in payload.split(',') if r.strip().strip("'\"")]
        return [r for r in rotas if r not in ('fim', 'fim_direto')]

    return []


class RouterAgent(BaseAgent):
    nome_agente: str = 'router'
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
        resultado = self._runnable.invoke(
            {'messages': self._mensagens_recentes(state)},
            config=self._config_interno(config)
        )
        return self._processar_resultado(resultado)

    async def arun(self, state: GraphState, config: RunnableConfig) -> dict:
        resultado = await self._runnable.ainvoke(
            {'messages': self._mensagens_recentes(state)},
            config=self._config_interno(config)
        )
        return self._processar_resultado(resultado)