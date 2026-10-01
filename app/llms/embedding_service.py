from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingService:
    def __init__(self, api_key: str, model: str = 'gemini-embedding-2-preview', dim: int = 768):
        self._client = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        self._dim = dim

    def gerar_embedding(self, texto: str) -> list[float]:
        return self._client.embed_query(texto, output_dimensionality=self._dim)

    def gerar_embeddings_batch(self, textos: list[str]) -> list[list[float]]:
        return self._client.embed_documents(textos, output_dimensionality=self._dim)