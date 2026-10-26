from functools import lru_cache
from typing import Optional

from fastembed import TextEmbedding

from app.llms.base import LLMService


class EmbeddingService(LLMService):
    nome = 'embedding_service'

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        dim: int = 768,
    ):
        embedding_model = TextEmbedding(model_name=model)
        super().__init__(embedding_model)
        self._dim = dim
        # perguntas repetidas (FAQ) não precisam de nova chamada de inferência
        self._embed_cached = lru_cache(maxsize=256)(self._embed_query)

    @property
    def dim(self) -> int:
        """Dimensão dos vetores gerados, usada para criar as collections do Qdrant."""
        return self._dim

    def _embed_query(self, texto: str) -> tuple[float, ...]:
        vetor = next(self._llm.embed([texto]))
        return tuple(float(x) for x in vetor)

    def gerar_embedding(self, texto: str) -> list[float]:
        return list(self._embed_cached(texto))

    def gerar_embeddings_batch(self, textos: list[str]) -> list[list[float]]:
        gerador = self._llm.embed(textos)
        return [[float(x) for x in vetor] for vetor in gerador]
