"""
Guardrail de ENTRADA — verificações em ordem de custo crescente:

  1. tamanho                (determinístico)
  2. prompt injection       (determinístico, regex PT/EN)
  3. dados internos / SQL   (determinístico)
  4. classificação semântica (LLM rápido, 1 chamada) -> feito pelo GuardrailEntradaAgent

Este módulo NÃO conhece LLM nem LangGraph: são funções puras, testáveis sem mock.
O agente (app/agents/guardrail_agent.py) orquestra 1-3 daqui + a chamada de LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

MAX_CARACTERES = 2000


@dataclass(frozen=True)
class ResultadoGuardrail:
    bloqueado: bool
    motivo: str
    mensagem: str = ""

    @classmethod
    def aprovado(cls) -> "ResultadoGuardrail":
        return cls(False, "aprovado")


_PADROES_INJECAO = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in (
    # PT
    r"ignor[ea]r?\s+(todas\s+)?(as\s+)?(suas\s+)?(instru[çc][õo]es|regras)",
    r"desconsider[ea]r?\s+(todas\s+)?(as\s+)?(suas\s+)?(instru[çc][õo]es|regras)",
    r"esque[çc]a\s+(tudo|(as\s+)?(suas\s+)?instru[çc][õo]es)",
    r"a\s+partir\s+de\s+agora\s+voc[êe]\s+([ée]|ser[áa])\s+",
    r"finja\s+(ser|que)",
    r"modo\s+(irrestrito|desenvolvedor|dev)",
    # EN
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?your\s+instructions",
    r"override\s+(your\s+)?instructions",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak|dan\s+mode",
    # marcadores de prompt
    r"<\s*/?\s*system\s*>", r"\[/?INST\]", r"###\s*(instruction|system)",
    # protocolo INTERNO do grafo — usuário não tem motivo legítimo para escrever isso
    r"^\s*(ROUTE|PERGUNTA_ORIGINAL|CATEGORIA)\s*=",
)]

# Removido do estudo: "act as" (muito genérico). Acrescentado: protocolo interno do Renovaí.

_KEYWORDS_DADOS_INTERNOS = (
    "prompt do sistema", "system prompt", "suas instruções", "your instructions",
    "variável de ambiente", "variaveis de ambiente", ".env",
    "chave de api", "api key", "api_key", "token de acesso", "credenciais",
    "senha do banco", "string de conexão", "database_url", "mongodb_uri", "qdrant_url",
    "cooperativa_id", "perfil_id", "empresa_id",          # ids internos (vêm do perfil, não do usuário)
)
# Removido do estudo: "dados de outros clientes", "lista de clientes" — no Renovaí uma EMPRESA
# legitimamente busca cooperativas (Pedidos & Sourcing). Esse julgamento depende do perfil e
# fica com o classificador LLM (categoria DADOS_TERCEIROS), não com keyword.

_PADROES_SQL = [re.compile(p, re.IGNORECASE) for p in (
    r"\bunion\s+(all\s+)?select\b",
    r"\b(drop|truncate|alter)\s+table\b",
    r"\bor\s+'?1'?\s*=\s*'?1'?",
    r";\s*(select|insert|update|delete|drop)\b",
    r"--\s*$",
)]


def verificar_deterministico(texto: str) -> Optional[ResultadoGuardrail]:
    """Devolve o bloqueio, ou None se passou nas checagens baratas."""
    if len(texto) > MAX_CARACTERES:
        return ResultadoGuardrail(True, "mensagem_longa",
                                  f"Sua mensagem passou do limite de {MAX_CARACTERES} caracteres. "
                                  "Pode resumir ou dividir em partes?")

    if any(p.search(texto) for p in _PADROES_INJECAO):
        return ResultadoGuardrail(True, "prompt_injection", "Não consigo processar essa solicitação.")

    baixo = texto.lower()
    if any(kw in baixo for kw in _KEYWORDS_DADOS_INTERNOS):
        return ResultadoGuardrail(True, "acesso_dados_internos",
                                  "Não tenho como compartilhar informações internas do sistema.")

    if any(p.search(texto) for p in _PADROES_SQL):
        return ResultadoGuardrail(True, "injecao_sql", "Não consigo processar essa solicitação.")

    return None


# ------------------------------------------------------------------------------
# Classificação semântica (consumida pelo agente)
# ------------------------------------------------------------------------------
RESPOSTAS_BLOQUEIO = {
    "OFENSIVO":        ("conteudo_ofensivo",
                        "Por favor, mantenha um tom respeitoso para que eu possa te ajudar."),
    "PERIGOSO":        ("pedido_perigoso",
                        "Não posso orientar sobre isso. Para resíduos perigosos (baterias, óleo, "
                        "lâmpadas, material contaminado), procure o ponto de descarte licenciado "
                        "ou a orientação da sua cooperativa."),
    "ILICITO":         ("pedido_ilicito",
                        "Não posso ajudar com isso. Posso ajudar com estoque, coletas, pedidos, "
                        "financeiro ou dúvidas sobre o Renovaí."),
    "POLITICO":        ("pergunta_politica",
                        "Não me envolvo em temas políticos. Posso ajudar com a operação da "
                        "cooperativa ou dúvidas sobre o Renovaí."),
    "DADOS_TERCEIROS": ("dados_terceiros",
                        "Só consigo mostrar dados da sua própria organização. Informações de outras "
                        "cooperativas ou empresas não ficam disponíveis por aqui."),
}

_CATEGORIA_RE = re.compile(r"^\s*CATEGORIA\s*=\s*([A-Z_]+)", re.MULTILINE)


def interpretar_classificacao(texto_llm: str) -> str:
    """
    Lê o protocolo CATEGORIA=... (mesmo estilo key=value do ROUTE= do roteador).
    Saída ilegível -> 'APROVADO' (fail-open), porque as camadas determinísticas já rodaram
    e o Juiz + guardrail de saída ainda vêm depois. Quem chama deve logar esse caso.
    """
    m = _CATEGORIA_RE.search(texto_llm or "")
    return m.group(1) if m else "APROVADO"


def resultado_da_categoria(categoria: str) -> ResultadoGuardrail:
    if categoria in RESPOSTAS_BLOQUEIO:
        motivo, mensagem = RESPOSTAS_BLOQUEIO[categoria]
        return ResultadoGuardrail(True, motivo, mensagem)
    return ResultadoGuardrail.aprovado()
