from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.factory import LLMFactory


class ResumoService:
    def __init__(self, llm_factory: LLMFactory, provider: str, tier: str):
        self._llm = llm_factory.get(provider, tier)
        self._PROMPT_RESUMO = """\
            Você é um assistente que resume conversas de acompanhamento de cooperativas e empresas recicladoras.
            Gere um resumo conciso em 2-4 frases capturando:
            - O que o usuário fez
            - O que o usuário perguntou
            - Informações relevantes mencionadas
    
            Responda APENAS com o resumo, sem introdução ou explicação.
    
            Conversa:
            {conversa}
        """
        self._embeddings = llm_factory.get(provider, tier, model='gemini-embedding-2-preview')
        self._EMBEDDING_DIM = 768

    def _formatar_conversa(self, mensagens: list[dict]) -> str:
        """Formata o array de mensagens em texto para o prompt de resumo."""
        linhas = []
        for msg in mensagens:
            linhas.append(f'{msg['role']}: {msg['content']}')
        return '\n'.join(linhas)

    def gerar_resumo(self, mensagens: List[Dict], resumo_anterior: Optional[str] = None) -> str:
        historico = self._formatar_conversa(mensagens)
        contexto_anterior = f'Resumo anterior: {resumo_anterior}\n\n' if resumo_anterior else ''

        resposta = self._llm.invoke([
                SystemMessage(content=self._PROMPT_RESUMO),
                HumanMessage(content=f'{contexto_anterior}Conversa:\n{historico}'),
        ])
        return resposta.content

    def gerar_embedding(self, texto: str) -> list[float]:
        """Gera um vetor de 768 dimensões para o texto informado."""
        return self._embeddings.embed_query(texto, output_dimensionality=self._EMBEDDING_DIM)

    def gerar_embeddings_batch(self, textos: list[str]) -> list[list[float]]:
        """Gera embeddings para uma lista de textos de uma vez (mais eficiente)."""
        return self._embeddings.embed_documents(textos, output_dimensionality=self._EMBEDDING_DIM)