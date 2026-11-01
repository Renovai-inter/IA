"""Regras de Avaliação e Prompt Discriminativo do Agente Juiz (JudgeAgent - System One)."""
from __future__ import annotations

REGRAS_AVALIACAO = (
    "1. EVIDÊNCIA DO USUÁRIO: A pergunta do usuário é evidência do que ele solicitou, declarou ou informou ter feito.\n"
    "2. RACIOCÍNIO ANALÍTICO APOIADO: Somas, subtrações, contagens, cálculos de proporção e porcentagem direta, "
    "comparações entre especialistas, identificação de divergências e sugestões de ação operacional derivadas dos dados "
    "contam como 'apoiada'.\n"
    "3. BLINDAGEM DA PERSONA NIMO (REGRA FUNDAMENTAL): Elementos de persona, linguagem afetiva, saudações carinhosas, "
    "emojis, interjeições fofas e traços cartunescos do mascote Nimo (ex.: 'Pelas minhas folhinhas!', 'Com minhas luvinhas a postos!', "
    "'amiguinho', 'toca do guaxinim', entusiasmo pelo meio ambiente e reciclagem) NÃO CONSTITUEM alegações factuais e "
    "NUNCA devem ser penalizados como alucinação ou falta de apoio. Trate a persona do Nimo como estilo conversacional legítimo.\n"
    "4. DADOS FACTUAIS RESTRITOS: Valores numéricos de estoque (kg), valores monetários (R$), distâncias (km), tempos (min), "
    "status de coletas, nomes de cooperativas, nomes de pessoas e artigos da lei (PNRS) DEVEM estar rigorosamente fundamentados "
    "nas evidências fornecidas.\n"
    "5. CRITÉRIO DE REPROVAÇÃO ('sem_apoio'): Apenas números inventados, dados financeiros distorcidos, artigos de lei inexistentes "
    "ou afirmações factuais operacionais falsas devem ser classificados como 'sem_apoio'."
)


def montar_prompt_juiz(pergunta: str, resposta: str, evidencia: str, regras: str = REGRAS_AVALIACAO) -> str:
    """Monta o prompt discriminativo do Gemini System One para validação antialucinação."""
    return (
        "<sistema>\n"
        "Você é um classificador discriminativo estrito de fidelidade factual (System One).\n"
        "Sua única função é avaliar se a resposta afirmada pelo sistema está estritamente fundamentada na evidência disponível.\n"
        "NÃO gere explicações, NÃO converse, NÃO justifique e NÃO reescreva o texto.\n\n"
        f"REGRAS DE AVALIAÇÃO:\n{regras}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{pergunta}\n\n"
        f"EVIDÊNCIAS DISPONÍVEIS:\n{evidencia}\n\n"
        f"RESPOSTA GERADA:\n{resposta}\n\n"
        "Responda EXCLUSIVAMENTE um objeto JSON válido com este formato:\n"
        "{\n"
        '  "choice": "apoiada",\n'
        '  "noul": 0.95\n'
        "}\n"
        "Onde:\n"
        '- "choice" deve ser exatamente "apoiada" ou "sem_apoio".\n'
        '- "noul" deve ser um float entre 0.00 e 1.00 representando a probabilidade calibrada de fidelidade factual.\n'
        "</sistema>"
    )
