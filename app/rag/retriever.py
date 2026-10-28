"""
Retriever do Agente RAG / FAQ Institucional e Normas Externas (PNRS).

Fica entre as tools do agente e os repositories do Qdrant:
- `FaqChunksRepository` (coleção `faq_chunks`): FAQ institucional do Renovaí.
- `normas_repository` (coleção `normas_residuos`): Legislação externa da Política Nacional de Resíduos Sólidos (Lei nº 12.305/2010).

O `FaqRetriever` decide COMO transformar a pergunta do usuário em busca vetorial,
permitindo consulta focada (apenas FAQ ou apenas Normas) ou consulta dual integrada
ordenada pela relevância semântica (score).
"""
from dataclasses import dataclass
from typing import List, Optional

from app.llms.embedding_service import EmbeddingService
from app.repository.qdrant.faq_chunks_repository import FaqChunksRepository


@dataclass(slots=True)
class FaqChunkCtx:
    conteudo: str
    fonte: str
    pagina: int = 0
    score: float = 0.0
    artigo: Optional[str] = None
    capitulo: Optional[str] = None
    tipo: str = 'faq'  # 'faq' ou 'norma'


class FaqRetriever:
    def __init__(
        self,
        repository: FaqChunksRepository,
        embedding_service: EmbeddingService,
        normas_repository: Optional[FaqChunksRepository] = None,
        k: int = 6,
    ):
        self._repository = repository
        self._embedding_service = embedding_service
        self._normas_repository = normas_repository
        self._k = k

    @property
    def repository(self) -> FaqChunksRepository:
        return self._repository

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def normas_repository(self) -> Optional[FaqChunksRepository]:
        return self._normas_repository

    @property
    def k(self) -> int:
        return self._k

    def buscar_faq(self, pergunta: str, k: Optional[int] = None) -> List[FaqChunkCtx]:
        """Busca os chunks institucionais do FAQ do Renovaí mais relevantes para a pergunta."""
        embedding = self._embedding_service.gerar_embedding(pergunta)
        resultados = self._repository.buscar(embedding, k or self._k)

        if not resultados:
            return []

        return [
            FaqChunkCtx(
                conteudo=resultado.item.page_content,
                fonte=resultado.item.source or 'FAQ Renovaí',
                pagina=resultado.item.page_number,
                score=resultado.score,
                artigo=resultado.item.artigo,
                capitulo=resultado.item.capitulo,
                tipo='faq',
            )
            for resultado in resultados
        ]

    def buscar_normas(self, pergunta: str, k: Optional[int] = None) -> List[FaqChunkCtx]:
        """Busca artigos e trechos na legislação externa (PNRS - Lei nº 12.305/2010)."""
        if not self._normas_repository:
            return []

        embedding = self._embedding_service.gerar_embedding(pergunta)
        resultados = self._normas_repository.buscar(embedding, k or self._k)

        if not resultados:
            return []

        return [
            FaqChunkCtx(
                conteudo=resultado.item.page_content,
                fonte=resultado.item.source or 'PNRS - Lei nº 12.305/2010',
                pagina=resultado.item.page_number,
                score=resultado.score,
                artigo=resultado.item.artigo,
                capitulo=resultado.item.capitulo,
                tipo='norma',
            )
            for resultado in resultados
        ]

    def buscar(self, pergunta: str, k: Optional[int] = None, fonte: Optional[str] = None) -> List[FaqChunkCtx]:
        """
        Busca os chunks mais relevantes para a pergunta recebida.

        Permite busca específica por fonte ('faq', 'normas') ou consulta dual
        consolidada (quando fonte é None ou 'todas').
        """
        if fonte == 'faq':
            return self.buscar_faq(pergunta, k)
        if fonte == 'normas':
            return self.buscar_normas(pergunta, k)

        limite = k or self._k
        if not self._normas_repository:
            return self.buscar_faq(pergunta, limite)

        res_faq = self.buscar_faq(pergunta, limite)
        res_normas = self.buscar_normas(pergunta, limite)

        combinados = sorted(res_faq + res_normas, key=lambda x: x.score, reverse=True)
        return combinados[:limite]
