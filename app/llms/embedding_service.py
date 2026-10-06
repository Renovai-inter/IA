from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.llms.base import LLMService


class EmbeddingService(LLMService):
    nome = 'embedding_service'

    def __init__(self, api_key: str, model: str = 'gemini-embedding-2-preview', dim: int = 768):
        super().__init__(GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key))
        self._dim = dim
        # perguntas repetidas (FAQ) não precisam de nova chamada de rede
        self._embed_cached = lru_cache(maxsize=256)(self._embed_query)

    def _embed_query(self, texto: str) -> tuple[float, ...]:
        return tuple(self._llm.embed_query(texto, output_dimensionality=self._dim))

    def gerar_embedding(self, texto: str) -> list[float]:
        return list(self._embed_cached(texto))

    def gerar_embeddings_batch(self, textos: list[str]) -> list[list[float]]:
        return self._llm.embed_documents(textos, output_dimensionality=self._dim)