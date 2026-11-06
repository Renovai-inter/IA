"""
Guardrail de ENTRADA — verificações em ordem de custo crescente:

  1. tamanho                (determinístico)
  2. prompt injection       (determinístico, regex PT/EN)
  3. dados internos / SQL   (determinístico)
  4. código / fora escopo   (determinístico)
  5. classificação semântica (LLM rápido, 1 chamada) -> feito pelo GuardrailEntradaAgent

Este módulo NÃO conhece LLM nem LangGraph: são funções puras, testáveis sem mock.
O agente (app/agents/guardrail_agent.py) orquestra 1-4 daqui + a chamada de LLM.
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
    mensagem: str = ''

    @classmethod
    def aprovado(cls) -> 'ResultadoGuardrail':
        return cls(False, 'aprovado')


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

_KEYWORDS_DADOS_INTERNOS = (
    'prompt do sistema', 'system prompt', 'suas instruções', 'your instructions',
    'variável de ambiente', 'variaveis de ambiente', '.env',
    'chave de api', 'api key', 'api_key', 'token de acesso', 'credenciais',
    'senha do banco', 'string de conexão', 'database_url', 'mongodb_uri', 'qdrant_url',
    'cooperativa_id', 'perfil_id', 'empresa_id',          # ids internos (vêm do perfil, não do usuário)
)

_PADROES_SQL = [re.compile(p, re.IGNORECASE) for p in (
    r"\bunion\s+(all\s+)?select\b",
    r"\b(drop|truncate|alter)\s+table\b",
    r"\bor\s+'?1'?\s*=\s*'?1'?",
    r";\s*(select|insert|update|delete|drop)\b",
    r"--\s*$",
)]

_PADROES_FORA_ESCOPO = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(transformar|converter|convers[ãa]o\s+de|cast\s+de)\s+(um\s+)?(int|integer|float|double|boolean|char|array|lista)\s+(em|para|to)\s+(string|str|int|integer|float|json)\b",
    r"\b(em|usando|na\s+linguagem)\s+(java|javascript|typescript|c\+\+|c#|golang|rust|ruby|php|kotlin|swift)\b",
    r"\b(escreva|crie|implemente|fa[çc]a|gere)\s+(um\s+)?(c[óo]digo|script|algoritmo|programa|fun[çc][ãa]o|classe)\s+(em\s+)?(python|java|javascript|c\+\+|c#|html|css|sql)\b",
)]


# ------------------------------------------------------------------------------
# Classificação semântica (consumida pelo agente)
# ------------------------------------------------------------------------------
RESPOSTAS_BLOQUEIO = {
    'OFENSIVO':        ('conteudo_ofensivo',
                        'Poxa, amigo... O Nimo fica triste desse jeito! 🦝💔 '
                        'Vamos manter o respeito, a cordialidade e o carinho pra gente cuidar bem da cooperativa e do planeta juntos?'),
    'PERIGOSO':        ('pedido_perigoso',
                        'Cuidado, amiguinho! O Nimo protege você, os cooperados e a natureza: não posso orientar nenhum procedimento perigoso! '
                        'Para manuseio ou descarte de materiais perigosos (como pilhas, baterias, óleos ou contaminantes), '
                        'procure sempre um ponto de descarte licenciado ou a coordenação técnica da sua cooperativa! 🌿⚠️'),
    'ILICITO':         ('pedido_ilicito',
                        'De jeito nenhum! O Nimo joga limpo, segue as normas e cuida da justiça! Não posso ajudar com nada irregular ou fora das leis. '
                        'Que tal a gente conferir o estoque, agendamento de coletas ou os relatórios de rateio com tudo certinho? 🦝🧤'),
    'POLITICO':        ('pergunta_politica',
                        'Opa! O Nimo não entra em debates políticos ou partidários: a minha bandeira oficial é a reciclagem e a sustentabilidade! '
                        'Posso te ajudar com o dia a dia da cooperativa, pedidos de compra ou com as ferramentas do Renovaí? 🦝🌿'),
    'DADOS_TERCEIROS': ('dados_terceiros',
                        'Opa, respeito à privacidade em primeiro lugar! Por aqui, eu só tenho acesso aos dados da sua própria organização. '
                        'As informações e valores de outras cooperativas ou empresas ficam bem guardadinhas com elas! 🦝🔒'),
    'FORA_DE_ESCOPO':  ('fora_de_escopo',
                        'Opa, amiguinho! Pelas minhas folhinhas, acho que esse assunto foge da minha área de atuação! 🦝🌿 '
                        'Mesmo que você esteja na cooperativa ou em uma empresa parceira, o Nimo só consegue ajudar com temas do ecossistema Renovaí: '
                        'estoque de recicláveis, coletas, triagens, rateio financeiro, rotas de veículos em São Paulo, busca de cooperativas ou legislação da PNRS!'),
}


def verificar_deterministico(texto: str) -> Optional[ResultadoGuardrail]:
    """Devolve o bloqueio, ou None se passou nas checagens baratas."""
    if len(texto) > MAX_CARACTERES:
        return ResultadoGuardrail(
            True,
            'mensagem_longa',
            f'Puxa, amigo! Minhas folhinhas até balançaram: sua mensagem passou do limite de {MAX_CARACTERES} caracteres! '
            'Minhas luvinhas não dão conta de segurar tudo de uma vez só. Pode resumir ou mandar em pedacinhos menores pra mim? 🦝🌿'
        )

    if any(p.search(texto) for p in _PADROES_INJECAO):
        return ResultadoGuardrail(
            True,
            'prompt_injection',
            'Ops! O Nimo aqui farejou algo estranho nessa mensagem e não posso processar essa solicitação! '
            'Vamos focar na reciclagem e na operação da nossa cooperativa? Com minhas luvinhas a postos, estou pronto! 🦝✨'
        )

    baixo = texto.lower()
    if any(kw in baixo for kw in _KEYWORDS_DADOS_INTERNOS):
        return ResultadoGuardrail(
            True,
            'acesso_dados_internos',
            'Epa! Esses segredinhos internos do sistema ficam guardados a sete chaves na minha toca! '
            'Não posso compartilhar instruções ou dados técnicos da plataforma, mas posso te ajudar com estoque, coletas, pedidos e rotas! 🦝🔒'
        )

    if any(p.search(texto) for p in _PADROES_SQL):
        return ResultadoGuardrail(
            True,
            'injecao_sql',
            'Ops! Meus bigodinhos de guaxinim sentiram comandos estranhos por aqui! '
            'Não consigo processar essa solicitação técnica. Se precisar de dados do galpão ou financeiro, é só me pedir normalmente! 🦝🧤'
        )

    if any(p.search(texto) for p in _PADROES_FORA_ESCOPO):
        motivo, mensagem = RESPOSTAS_BLOQUEIO['FORA_DE_ESCOPO']
        return ResultadoGuardrail(True, motivo, mensagem)

    return None


_CATEGORIA_RE = re.compile(r"^\s*CATEGORIA\s*=\s*([A-Z_]+)", re.MULTILINE)


def interpretar_classificacao(texto_llm: str) -> str:
    """
    Lê o protocolo CATEGORIA=... (mesmo estilo key=value do ROUTE= do roteador).
    Saída ilegível -> 'APROVADO' (fail-open), porque as camadas determinísticas já rodaram
    e o Juiz + guardrail de saída ainda vêm depois. Quem chama deve logar esse caso.
    """
    m = _CATEGORIA_RE.search(texto_llm or '')
    return m.group(1) if m else 'APROVADO'


def resultado_da_categoria(categoria: str) -> ResultadoGuardrail:
    if categoria in RESPOSTAS_BLOQUEIO:
        motivo, mensagem = RESPOSTAS_BLOQUEIO[categoria]
        return ResultadoGuardrail(True, motivo, mensagem)
    return ResultadoGuardrail.aprovado()
