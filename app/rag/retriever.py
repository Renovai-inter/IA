"""
Retriever do Agente RAG / FAQ Institucional (#3 na arquitetura alvo).

Fica entre a tool do agente e o `FaqChunksRepository`: o repository é camada
mecânica — só sabe conversar com a collection `faq_chunks` do Qdrant, mesmo
papel que `repository/mongodb/sessao_repository.py` e
`repository/qdrant/resumo_repository.py` cumprem para os `MemoryStore`s (não
decide nada, só executa a query que mandarem). O `FaqRetriever` é quem decide
COMO transformar a pergunta do usuário em busca — gerar o embedding, aplicar
o `k` — e devolve um contexto de domínio (`FaqChunkCtx`) já pronto pra
tool/agente citar a fonte, o mesmo papel que `MemoriaCtx` cumpre para os
`MemoryStore`s: a tool recebe o retriever (não o repository) e só enxerga
`FaqChunkCtx`, nunca o payload bruto do Qdrant.
"""
from typing import List, Optional

from pydantic import BaseModel

from app.llms.resumo_service import ResumoService
from app.repository.qdrant.faq_chunks_repository import FaqChunksRepository


class FaqChunkCtx(BaseModel):
    conteudo: str
    fonte: str
    pagina: int
    score: float


class FaqRetriever:
    def __init__(self, repository: FaqChunksRepository, resumo_service: ResumoService, k: int = 6):
        self.repository = repository
        self.resumo_service = resumo_service
        self.k = k

    def buscar(self, pergunta: str, k: Optional[int] = None) -> List[FaqChunkCtx]:
        """
        Busca os chunks institucionais mais relevantes para a pergunta recebida.

        Devolve lista vazia quando não há resultado — quem decide o que fazer
        com "nada encontrado" (responder que não sabe, escalar, etc.) é a
        tool/agente, não o retriever.
        """
        embedding = self.resumo_service.gerar_embedding(pergunta)
        resultados = self.repository.buscar_faq(embedding, k or self.k)

        if not resultados:
            return []

        return [
            FaqChunkCtx(
                conteudo=resultado.item.page_content,
                fonte=resultado.item.source,
                pagina=resultado.item.page_number,
                score=resultado.score,
            )
            for resultado in resultados
        ]
