from functools import lru_cache
import logging
import os
from typing import Any, Optional

import httpx

from app.llms.base import LLMService

log = logging.getLogger('renovai.embeddings')

HF_INFERENCE_URL = 'https://router.huggingface.co/hf-inference/models'


class EmbeddingService(LLMService):
    nome = 'embedding_service'

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        dim: int = 768,
        gemini_api_key: Optional[str] = None,
    ):
        # Sempre inicializa com None para garantir 0 MB de modelos locais em RAM
        super().__init__(None)
        self._dim = dim
        self._model = model
        self._hf_token = (api_key or os.getenv('HF_TOKEN') or '').strip() or None
        self._gemini_api_key = (gemini_api_key or os.getenv('GEMINI_API_KEY') or '').strip() or None
        self._api_url = f'{HF_INFERENCE_URL}/{model}/pipeline/feature-extraction'
        self._gemini_embeddings: Any = None
        # Prioriza o Google text-embedding-004 (alta velocidade, estável e gratuito)
        self._use_gemini = bool(self._gemini_api_key)
        self._use_hf = bool(self._hf_token) and not self._use_gemini

        if self._use_gemini:
            log.info('[embedding_service] Usando Google text-embedding-004 como motor primário de embeddings (dim=768).')
        elif self._use_hf:
            log.info(f'[embedding_service] Usando Hugging Face Inference API ({model}).')
        else:
            log.info('[embedding_service] Nenhuma chave remota detectada inicialmente.')

        # perguntas repetidas (FAQ) não precisam de nova chamada de inferência
        self._embed_cached = lru_cache(maxsize=128)(self._embed_query)

    def _obter_gemini_embeddings(self, model_name: str = 'gemini-embedding-2-preview') -> Any:
        if self._gemini_embeddings is None or getattr(self._gemini_embeddings, 'model', None) != model_name:
            if not self._gemini_api_key:
                raise RuntimeError(
                    '[embedding_service] Nem HF_TOKEN nem GEMINI_API_KEY estão configurados para gerar embeddings via API.'
                )
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._gemini_embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self._gemini_api_key,
            )
        return self._gemini_embeddings

    @property
    def dim(self) -> int:
        """Dimensão dos vetores gerados, usada para criar as collections do Qdrant."""
        return self._dim

    def _embed_query(self, texto: str) -> tuple[float, ...]:
        if self._use_gemini:
            try:
                gemini_emb = self._obter_gemini_embeddings('gemini-embedding-2-preview')
                vetor = gemini_emb.embed_query(texto, output_dimensionality=self._dim)
                return tuple(float(x) for x in vetor)
            except Exception as exc:
                log.warning(
                    f"[embedding_service] Falha com gemini-embedding-2-preview ({exc}). Tentando text-embedding-004..."
                )
                try:
                    gemini_emb = self._obter_gemini_embeddings('text-embedding-004')
                    vetor = gemini_emb.embed_query(texto, output_dimensionality=self._dim)
                    return tuple(float(x) for x in vetor)
                except Exception as exc2:
                    log.warning(
                        f"[embedding_service] Falha em text-embedding-004 ({exc2}). "
                        "Tentando alternativa Hugging Face se disponível."
                    )
                    self._use_gemini = False

        if self._use_hf:
            try:
                vetor = self._chamar_hf_api(texto)
                if vetor and isinstance(vetor[0], list):
                    vetor = vetor[0]
                return tuple(float(x) for x in vetor)
            except Exception as exc:
                log.warning(
                    f"[embedding_service] Falha na Inference API do Hugging Face ({exc})."
                )
                self._use_hf = False

        if self._llm is not None:
            vetor = next(self._llm.embed([texto]))
            return tuple(float(x) for x in vetor)

        # Última tentativa via Gemini
        gemini_emb = self._obter_gemini_embeddings()
        vetor = gemini_emb.embed_query(texto, output_dimensionality=self._dim)
        return tuple(float(x) for x in vetor)

    def _chamar_hf_api(self, payload: str | list[str]) -> list[float] | list[list[float]]:
        headers = {
            'Authorization': f'Bearer {self._hf_token}',
            'Content-Type': 'application/json',
        }
        body = {
            'inputs': payload,
            'options': {'wait_for_model': True, 'use_cache': True},
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(self._api_url, headers=headers, json=body)
            if resp.status_code == 401:
                raise ValueError(
                    '[embedding_service] Erro 401 no Hugging Face: HF_TOKEN inválido ou não autorizado.'
                )
            resp.raise_for_status()
            data = resp.json()

        return self._normalizar_resposta_hf(data, is_batch=isinstance(payload, list))

    @staticmethod
    def _normalizar_resposta_hf(data: Any, is_batch: bool) -> list[float] | list[list[float]]:
        # Caso 1: Entrada única (str)
        if not is_batch:
            if not isinstance(data, list):
                return [float(x) for x in data]
            if isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            if isinstance(data[0], list):
                if isinstance(data[0][0], (int, float)):
                    if len(data) == 1:
                        return [float(x) for x in data[0]]
                    # Média dos tokens se vier sem pooling
                    return [float(sum(col) / len(col)) for col in zip(*data)]
            return [float(x) for x in data]

        # Caso 2: Lote de textos (list[str])
        if not isinstance(data, list):
            return [[float(x) for x in data]]
        # 2D: [[v1, v2...], [v1, v2...]]
        if isinstance(data[0], list) and (not data[0] or isinstance(data[0][0], (int, float))):
            return [[float(x) for x in item] for item in data]
        # 3D: [tokens_doc1, tokens_doc2] -> aplica pooling
        resultado = []
        for doc in data:
            if isinstance(doc, list) and doc and isinstance(doc[0], list):
                pooled = [float(sum(col) / len(col)) for col in zip(*doc)]
                resultado.append(pooled)
            else:
                resultado.append([float(x) for x in doc])
        return resultado

    def gerar_embedding(self, texto: str) -> list[float]:
        return list(self._embed_cached(texto))

    def gerar_embeddings_batch(self, textos: list[str]) -> list[list[float]]:
        if not textos:
            return []
        if self._use_gemini:
            try:
                gemini_emb = self._obter_gemini_embeddings('gemini-embedding-2-preview')
                vetores = gemini_emb.embed_documents(textos, output_dimensionality=self._dim)
                return [[float(x) for x in vetor] for vetor in vetores]
            except Exception as exc:
                log.warning(
                    f"[embedding_service] Falha no batch com gemini-embedding-2-preview ({exc}). Tentando text-embedding-004..."
                )
                try:
                    gemini_emb = self._obter_gemini_embeddings('text-embedding-004')
                    vetores = gemini_emb.embed_documents(textos, output_dimensionality=self._dim)
                    return [[float(x) for x in vetor] for vetor in vetores]
                except Exception as exc2:
                    log.warning(
                        f"[embedding_service] Falha no batch de text-embedding-004 ({exc2})."
                    )
                    self._use_gemini = False

        if self._use_hf:
            try:
                res = self._chamar_hf_api(textos)
                if res and isinstance(res[0], (int, float)):
                    return [res]  # type: ignore
                return res  # type: ignore
            except Exception as exc:
                log.warning(
                    f"[embedding_service] Falha no batch do Hugging Face ({exc})."
                )
                self._use_hf = False

        if self._llm is not None:
            gerador = self._llm.embed(textos)
            return [[float(x) for x in vetor] for vetor in gerador]

        gemini_emb = self._obter_gemini_embeddings()
        vetores = gemini_emb.embed_documents(textos, output_dimensionality=self._dim)
        return [[float(x) for x in vetor] for vetor in vetores]
