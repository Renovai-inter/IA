"""
Guardrail de SAÍDA — nunca bloqueia o turno; sempre devolve um texto entregável.

  1. vazamento de protocolo interno (ROUTE=, JSON cru do especialista)  (determinístico)
  2. PII gerada pelo modelo + tokens da entrada                         (determinístico)
  3. revisão de política por LLM — SÓ nas rotas configuradas            (agente)

No Renovaí a resposta carrega kg, saldos e valores de rateio vindos do Postgres
quem valida conteúdo é o Juiz. Aqui o LLM só emite um VEREDITO, e a ação é
determinística (anexar aviso ou trocar por mensagem segura).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from app.guardrails.pii import MapaPII, redigir, resolver_tokens

MENSAGEM_FALHA_FORMATACAO = (
    "Não consegui montar a resposta agora. Pode repetir a pergunta, de outra forma se possível?"
)

_PROTOCOLO_INTERNO = re.compile(
    r"^\s*[\"']?(ROUTE|ROUTES|PERGUNTA_ORIGINAL|CATEGORIA|VEREDITO)[\"']?\s*[:=]",
    re.MULTILINE,
)


def _parece_json_de_especialista(texto: str) -> bool:
    bruto = texto.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not (bruto.startswith("{") and bruto.endswith("}")):
        return False
    try:
        dados = json.loads(bruto)
        if isinstance(dados, dict):
            return any(k in dados for k in ("dominio", "ROUTE", "ROUTES", "route", "routes", "PERGUNTA_ORIGINAL"))
        return False
    except ValueError:
        return False


def higienizar(texto: str, mapa_pii: MapaPII) -> tuple[str, Optional[str]]:
    """
    Devolve (texto_final, motivo_de_correcao | None).
    """
    if _PROTOCOLO_INTERNO.search(texto) or _parece_json_de_especialista(texto):
        return MENSAGEM_FALHA_FORMATACAO, "vazamento_protocolo"

    limpo = resolver_tokens(redigir(texto), mapa_pii)
    return limpo, ("pii_redigida" if limpo != texto else None)


# ------------------------------------------------------------------------------
# Veredito do LLM (consumido pelo agente)
# ------------------------------------------------------------------------------
AVISO_PROMESSA_FINANCEIRA = (
    "\n\n_Valores calculados a partir dos registros atuais do sistema. O valor final de "
    "rateio/pagamento depende do fechamento do período e da confirmação da cooperativa._"
)

MENSAGEM_ORIENTACAO_PERIGOSA = (
    "Não consigo orientar esse procedimento por aqui. Para manuseio ou descarte de resíduos "
    "perigosos, siga as normas da sua cooperativa e procure um ponto de descarte licenciado."
)

_VEREDITO_RE = re.compile(r"^\s*VEREDITO\s*=\s*([A-Z_]+)", re.MULTILINE)


def interpretar_veredito(texto_llm: str) -> str:
    m = _VEREDITO_RE.search(texto_llm or "")
    return m.group(1) if m else "OK"


def aplicar_veredito(texto: str, veredito: str) -> tuple[str, Optional[str]]:
    if veredito == "PROMESSA_FINANCEIRA":
        return texto + AVISO_PROMESSA_FINANCEIRA, "aviso_financeiro"
    if veredito == "ORIENTACAO_PERIGOSA":
        return MENSAGEM_ORIENTACAO_PERIGOSA, "orientacao_perigosa"
    return texto, None
