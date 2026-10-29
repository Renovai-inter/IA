"""
Script de ingestão de bases RAG no Qdrant:
1. FAQ institucional do Renovaí (coleção `faq_chunks`).
2. Legislação externa da Política Nacional de Resíduos Sólidos (PNRS / Lei nº 12.305/2010) (coleção `normas_residuos`).

Uso:
    python -m app.rag.ingestion all   # Ingestão de FAQ e PNRS
    python -m app.rag.ingestion faq   # Apenas FAQ
    python -m app.rag.ingestion pnrs  # Apenas PNRS
"""

from pathlib import Path
import re
from typing import Optional
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from app.core.config import Settings
from app.repository.qdrant.db import QdrantConnectionFactory
from app.llms.embedding_service import EmbeddingService


_CHUNK_SIZE    = 700
_CHUNK_OVERLAP = 150
_BATCH_SIZE    = 25

_settings: Optional[Settings] = None
_qdrant_factory: Optional[QdrantConnectionFactory] = None
_qdrant = None
_embedding_service: Optional[EmbeddingService] = None


def _obter_recursos():
    global _settings, _qdrant_factory, _qdrant, _embedding_service
    if _settings is None:
        _settings = Settings()
        _qdrant_factory = QdrantConnectionFactory(_settings.QDRANT_URL, _settings.QDRANT_API_KEY)
        _qdrant = _qdrant_factory.connect()
        _embedding_service = EmbeddingService(
            _settings.HF_TOKEN,
            gemini_api_key=_settings.GEMINI_API_KEY,
        )
    return _settings, _qdrant_factory, _qdrant, _embedding_service


def _garantir_collection(collection_name: str, dim: int) -> None:
    """Garante que a collection exista no Qdrant com a dimensão configurada."""
    _, _, qdrant, _ = _obter_recursos()
    if not qdrant.collection_exists(collection_name):
        print(f'[ingest] Criando collection {collection_name} (dim={dim})...')
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        info = qdrant.get_collection(collection_name)
        params = info.config.params.vectors
        size_existente = getattr(params, 'size', None)
        if isinstance(params, dict) and '' in params:
            size_existente = params[''].size
        if size_existente and size_existente != dim:
            print(f'[ingest] Dimensão alterada ({size_existente} -> {dim}). Recriando {collection_name}...')
            qdrant.delete_collection(collection_name)
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )


def _limpar_collection(collection_name: str) -> None:
    """Apaga os pontos existentes na collection antes da nova ingestão."""
    _, _, qdrant, embedding_service = _obter_recursos()
    _garantir_collection(collection_name, embedding_service.dim)
    info = qdrant.get_collection(collection_name)
    if info.points_count > 0:
        print(f'[ingest] Limpando {info.points_count} ponto(s) existente(s) de {collection_name}...')
        qdrant.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[])
            ),
        )


def _extrair_chunks_pnrs(caminho_pnrs: Path) -> list[dict]:
    """Segmenta a lei da PNRS por Artigos e Capítulos com enriquecimento de contexto."""
    text = caminho_pnrs.read_text(encoding='utf-8')
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    current_titulo = ''
    current_capitulo = ''
    current_artigo = 'Preâmbulo'
    current_artigo_lines = []

    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=['\n\n', '\n', '; ', '. ', ' '],
    )

    def _flush_artigo(art: str, tit: str, cap: str, lines_list: list[str]) -> None:
        content = '\n'.join(lines_list).strip()
        if not content:
            return
        prefix = f'PNRS (Lei nº 12.305/2010) - {cap}\n' if cap else 'PNRS (Lei nº 12.305/2010)\n'
        if len(content) <= 800:
            chunks.append({
                'page_content': prefix + content,
                'source': 'PNRS - Lei nº 12.305/2010',
                'artigo': art,
                'titulo': tit,
                'capitulo': cap,
                'page_number': 0,
            })
        else:
            for sc in splitter.split_text(content):
                chunks.append({
                    'page_content': prefix + sc,
                    'source': 'PNRS - Lei nº 12.305/2010',
                    'artigo': art,
                    'titulo': tit,
                    'capitulo': cap,
                    'page_number': 0,
                })

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('TÍTULO'):
            next_line = lines[i+1] if i+1 < len(lines) and not lines[i+1].startswith(('TÍTULO', 'CAPÍTULO', 'Art.')) else ''
            current_titulo = f'{line} - {next_line}' if next_line else line
            if next_line:
                i += 1
        elif line.startswith('CAPÍTULO'):
            next_line = lines[i+1] if i+1 < len(lines) and not lines[i+1].startswith(('TÍTULO', 'CAPÍTULO', 'Art.')) else ''
            current_capitulo = f'{line} - {next_line}' if next_line else line
            if next_line:
                i += 1
        else:
            m_art = re.match(r'^(Art\.\s*\d+º?)', line)
            if m_art:
                _flush_artigo(current_artigo, current_titulo, current_capitulo, current_artigo_lines)
                current_artigo = m_art.group(1)
                current_artigo_lines = [line]
            else:
                current_artigo_lines.append(line)
        i += 1

    _flush_artigo(current_artigo, current_titulo, current_capitulo, current_artigo_lines)
    return chunks


# Alias para testes existentes
extrair_chunks_pnrs = _extrair_chunks_pnrs


def ingerir_faq(faq_path: Optional[str] = None) -> int:
    """Indexa o FAQ oficial do Renovaí no Qdrant. Retorna o número de chunks inseridos."""
    settings, _, qdrant, embedding_service = _obter_recursos()
    caminho = Path(faq_path or settings.FAQ_PATH)
    print(f'[ingest] Carregando FAQ: {caminho}')
    conteudo_faq = caminho.read_text(encoding='utf-8')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([conteudo_faq], metadatas=[{'source': 'FAQ Renovaí'}])
    print(f'[ingest] {len(chunks)} chunk(s) gerado(s) para o FAQ')

    _limpar_collection(settings.QDRANT_COLLECTION_FAQ)

    textos = [chunk.page_content for chunk in chunks]

    for i in range(0, len(textos), _BATCH_SIZE):
        lote_textos = textos[i : i + _BATCH_SIZE]
        lote_chunks = chunks[i : i + _BATCH_SIZE]

        print(f'[ingest:faq] Gerando embeddings para chunk {i+1}-{i+len(lote_textos)}...')
        vetores = embedding_service.gerar_embeddings_batch(lote_textos)

        pontos = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vetor,
                payload={
                    'page_content': chunk.page_content,
                    'page_number':  chunk.metadata.get('page', 0),
                    'source':       str(chunk.metadata.get('source', 'FAQ Renovaí')),
                    'artigo':       None,
                    'titulo':       'FAQ Institucional Renovaí',
                    'capitulo':     None,
                },
            )
            for vetor, chunk in zip(vetores, lote_chunks)
        ]

        qdrant.upsert(collection_name=settings.QDRANT_COLLECTION_FAQ, points=pontos)

    print(f'[ingest:faq] Concluído! {len(chunks)} chunk(s) indexado(s) no Qdrant.')
    return len(chunks)


def ingerir_pnrs(pnrs_path: Optional[str] = None) -> int:
    """Indexa a legislação PNRS (Lei nº 12.305/2010) na coleção normas_residuos do Qdrant."""
    settings, _, qdrant, embedding_service = _obter_recursos()
    caminho = Path(pnrs_path or settings.PNRS_PATH)
    print(f'[ingest] Carregando PNRS: {caminho}')
    chunks = _extrair_chunks_pnrs(caminho)
    print(f'[ingest] {len(chunks)} chunk(s) semântico(s) gerado(s) para a PNRS')

    _limpar_collection(settings.QDRANT_COLLECTION_NORMAS)

    textos = [c['page_content'] for c in chunks]

    for i in range(0, len(textos), _BATCH_SIZE):
        lote_textos = textos[i : i + _BATCH_SIZE]
        lote_chunks = chunks[i : i + _BATCH_SIZE]

        print(f'[ingest:pnrs] Gerando embeddings para chunk {i+1}-{i+len(lote_textos)}...')
        vetores = embedding_service.gerar_embeddings_batch(lote_textos)

        pontos = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vetor,
                payload={
                    'page_content': chunk['page_content'],
                    'page_number':  chunk['page_number'],
                    'source':       chunk['source'],
                    'artigo':       chunk['artigo'],
                    'titulo':       chunk['titulo'],
                    'capitulo':     chunk['capitulo'],
                },
            )
            for vetor, chunk in zip(vetores, lote_chunks)
        ]

        qdrant.upsert(collection_name=settings.QDRANT_COLLECTION_NORMAS, points=pontos)

    print(f'[ingest:pnrs] Concluído! {len(chunks)} chunk(s) indexado(s) no Qdrant na coleção {settings.QDRANT_COLLECTION_NORMAS}.')
    return len(chunks)


def ingerir_tudo() -> dict[str, int]:
    """Indexa tanto o FAQ quanto a PNRS."""
    total_faq = ingerir_faq()
    total_pnrs = ingerir_pnrs()
    return {'faq': total_faq, 'pnrs': total_pnrs}


if __name__ == '__main__':
    import sys
    alvo = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    _, qdrant_factory, qdrant, _ = _obter_recursos()
    try:
        if alvo == 'faq':
            ingerir_faq()
        elif alvo == 'pnrs':
            ingerir_pnrs()
        else:
            ingerir_tudo()
    finally:
        qdrant_factory.close(qdrant)
