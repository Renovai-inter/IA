from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.rag.ingestion import extrair_chunks_pnrs
from app.rag.retriever import FaqChunkCtx, FaqRetriever
from app.repository.qdrant.faq_chunks_repository import FaqChunksPayload, ResultadoBusca
from app.tools.faq_chunks_tools import FaqToolkit


def _mock_resultado(conteudo: str, fonte: str, score: float, artigo: str | None = None) -> ResultadoBusca[FaqChunksPayload]:
    return ResultadoBusca(
        id='uuid-teste',
        score=score,
        item=FaqChunksPayload(
            page_content=conteudo,
            page_number=1,
            source=fonte,
            artigo=artigo,
            titulo='Título Teste',
            capitulo='Capítulo Teste',
        ),
    )


def test_faq_retriever_busca_faq_e_normas():
    mock_faq_repo = MagicMock()
    mock_normas_repo = MagicMock()
    mock_embedding_service = MagicMock()
    mock_embedding_service.gerar_embedding.return_value = [0.1] * 768

    mock_faq_repo.buscar.return_value = [
        _mock_resultado('Como cadastrar cooperado', 'FAQ Renovaí', 0.85)
    ]
    mock_normas_repo.buscar.return_value = [
        _mock_resultado('Art. 9º Ordem de prioridade', 'PNRS - Lei nº 12.305/2010', 0.95, artigo='Art. 9º')
    ]

    retriever = FaqRetriever(
        repository=mock_faq_repo,
        embedding_service=mock_embedding_service,
        normas_repository=mock_normas_repo,
        k=4,
    )

    # 1. Busca específica no FAQ
    res_faq = retriever.buscar_faq('cadastro')
    assert len(res_faq) == 1
    assert res_faq[0].fonte == 'FAQ Renovaí'
    assert res_faq[0].tipo == 'faq'

    # 2. Busca específica nas Normas
    res_normas = retriever.buscar_normas('ordem de prioridade')
    assert len(res_normas) == 1
    assert res_normas[0].artigo == 'Art. 9º'
    assert res_normas[0].fonte == 'PNRS - Lei nº 12.305/2010'
    assert res_normas[0].tipo == 'norma'

    # 3. Busca Dual consolidada ordenada por score
    res_dual = retriever.buscar('resíduos e cadastro')
    assert len(res_dual) == 2
    # Score 0.95 (normas) deve vir antes de 0.85 (faq)
    assert res_dual[0].score == 0.95
    assert res_dual[0].artigo == 'Art. 9º'
    assert res_dual[1].score == 0.85


def test_faq_retriever_sem_normas_repo():
    mock_faq_repo = MagicMock()
    mock_embedding_service = MagicMock()
    mock_embedding_service.gerar_embedding.return_value = [0.1] * 768
    mock_faq_repo.buscar.return_value = [
        _mock_resultado('FAQ puro', 'FAQ Renovaí', 0.8)
    ]

    retriever = FaqRetriever(
        repository=mock_faq_repo,
        embedding_service=mock_embedding_service,
        normas_repository=None,
    )

    res_normas = retriever.buscar_normas('lei')
    assert res_normas == []

    res_dual = retriever.buscar('pergunta')
    assert len(res_dual) == 1
    assert res_dual[0].conteudo == 'FAQ puro'


def test_faq_toolkit_tools_executam_corretamente():
    mock_retriever = MagicMock()
    mock_retriever.buscar_faq.return_value = [
        FaqChunkCtx(
            conteudo='Horário de funcionamento das 8h às 18h.',
            fonte='FAQ Renovaí',
            pagina=1,
            score=0.88,
        )
    ]
    mock_retriever.buscar_normas.return_value = [
        FaqChunkCtx(
            conteudo='Art. 33 São obrigados a estruturar e implementar sistemas de logística reversa...',
            fonte='PNRS - Lei nº 12.305/2010',
            pagina=0,
            score=0.92,
            artigo='Art. 33',
            tipo='norma',
        )
    ]

    toolkit = FaqToolkit(mock_retriever)
    tools = toolkit.get_tools()
    tool_names = [t.name for t in tools]
    assert 'faq_retriever' in tool_names
    assert 'normas_retriever' in tool_names

    # Teste tool faq_retriever
    resp_faq = toolkit.faq_retriever('horario')
    assert resp_faq['status'] == 'ok'
    assert 'Horário de funcionamento' in resp_faq['chunks'][0]['conteudo']

    # Teste tool normas_retriever
    resp_normas = toolkit.normas_retriever('logistica reversa')
    assert resp_normas['status'] == 'ok'
    assert resp_normas['chunks'][0]['artigo'] == 'Art. 33'
    assert 'logística reversa' in resp_normas['chunks'][0]['conteudo']

    # Teste com pergunta vazia
    assert toolkit.faq_retriever('')['status'] == 'error'
    assert toolkit.normas_retriever('  ')['status'] == 'error'


def test_extrair_chunks_pnrs_extrai_artigos_reais():
    pnrs_path = Path('data/PNRS.txt')
    if not pnrs_path.exists():
        pytest.skip('Arquivo data/PNRS.txt não encontrado')

    chunks = extrair_chunks_pnrs(pnrs_path)
    assert len(chunks) > 50

    # Verifica se os artigos emblemáticos estão presentes
    artigos = {c['artigo'] for c in chunks}
    assert 'Art. 1º' in artigos
    assert 'Art. 3º' in artigos
    assert 'Art. 9º' in artigos
    assert 'Art. 33º' in artigos or 'Art. 33' in artigos

    # Verifica metadados de fonte
    for chunk in chunks[:10]:
        assert chunk['source'] == 'PNRS - Lei nº 12.305/2010'
        assert chunk['page_content'].strip()
