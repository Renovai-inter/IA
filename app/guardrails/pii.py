"""
Anonimização de PII — camada determinística do Guardrail (#1).

Roda na api/routes/chat.py, antes de qualquer gravação:
  - o MongoDB (`sessao`) e o ResumoService só recebem texto anonimizado;
  - o grafo inteiro (roteador, especialistas, orquestrador) nunca vê o dado cru.

  - PLACA (perfil Motorista) e CEP (01310-100) ou precedido de "CEP".
  - CPF e CNPJ são validados pelo dígito verificador: 11/14 dígitos soltos que não são
    documento (ex.: um telefone sem máscara) não viram token.
  - O mesmo valor recebe o MESMO token dentro do turno, para o LLM entender que
    duas menções são a mesma pessoa/empresa.
"""
from __future__ import annotations

import re
import uuid
from typing import Callable, Dict, List, Optional, Tuple

MapaPII = Dict[str, str]


def _cpf_valido(valor: str) -> bool:
    d = re.sub(r"\D", "", valor)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for n in (9, 10):
        soma = sum(int(d[i]) * (n + 1 - i) for i in range(n))
        dv = (soma * 10) % 11 % 10
        if dv != int(d[n]):
            return False
    return True


def _cnpj_valido(valor: str) -> bool:
    d = re.sub(r"\D", "", valor)
    if len(d) != 14 or d == d[0] * 14:
        return False
    for n, pesos in ((12, [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]),
                     (13, [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])):
        resto = sum(int(d[i]) * pesos[i] for i in range(n)) % 11
        dv = 0 if resto < 2 else 11 - resto
        if dv != int(d[n]):
            return False
    return True


def _luhn_valido(valor: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", valor)][::-1]
    total = sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2) for i, x in enumerate(d))
    return total % 10 == 0


PADROES_PII: List[Tuple[str, re.Pattern, Optional[Callable[[str], bool]]]] = [
    ("EMAIL",    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), None),
    ("CARTAO",   re.compile(r"(?<!\d)\d{4}[ .-]?\d{4}[ .-]?\d{4}[ .-]?\d{4}(?!\d)"), _luhn_valido),
    ("CNPJ",     re.compile(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)"), _cnpj_valido),
    ("CPF",      re.compile(r"(?<![\d./])\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?![\d/])"), _cpf_valido),
    ("TELEFONE", re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}(?!\d)"), None),
    ("CEP",      re.compile(r"(?<!\d)\d{5}-\d{3}(?!\d)|\bCEP\s*:?\s*\d{5}-?\d{3}(?!\d)", re.IGNORECASE), None),
    ("PLACA",    re.compile(r"\b[A-Z]{3}-?\d[A-Z0-9]\d{2}\b", re.IGNORECASE), None),
]

_TOKEN_RE = re.compile(r"\[PII_([A-Z]+)_[0-9a-f]{6}\]")


def anonimizar(texto: str) -> Tuple[str, MapaPII]:
    """Troca PII por tokens [PII_<TIPO>_<hex>]. Devolve (texto, mapa token->valor)."""
    mapa: MapaPII = {}
    reverso: Dict[str, str] = {}

    for tipo, padrao, validador in PADROES_PII:
        def _trocar(m: re.Match) -> str:
            valor = m.group(0)
            if validador and not validador(valor):
                return valor
            if valor not in reverso:
                token = f"[PII_{tipo}_{uuid.uuid4().hex[:6]}]"
                reverso[valor] = token
                mapa[token] = valor
            return reverso[valor]
        texto = padrao.sub(_trocar, texto)

    return texto, mapa


def redigir(texto: str) -> str:
    """Remove PII que o MODELO tenha gerado (não veio do usuário)."""
    for tipo, padrao, validador in PADROES_PII:
        texto = padrao.sub(
            lambda m, t=tipo, v=validador: m.group(0) if (v and not v(m.group(0))) else f"[{t} OMITIDO]",
            texto,
        )
    return texto


def resolver_tokens(texto: str, mapa: MapaPII, restaurar: bool = False) -> str:
    """
    Resolve tokens na saída. Por padrão OMITE (não repete dado pessoal ao usuário).
    Tokens sem entrada no mapa (ex.: vindos de turnos anteriores) também são omitidos.
    """
    def _trocar(m: re.Match) -> str:
        token, tipo = m.group(0), m.group(1)
        if restaurar and token in mapa:
            return mapa[token]
        return f"[{tipo} OMITIDO]"
    return _TOKEN_RE.sub(_trocar, texto)
