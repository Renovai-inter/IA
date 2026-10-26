from langchain_core.tools import StructuredTool

from app.rag.retriever import FaqRetriever
from app.repository.qdrant.faq_chunks_repository import FaqChunksRepository
from app.tools.base import Toolkit


class FaqToolkit(Toolkit):
    nome = 'faq_toolkit'
    descricao = 'Tools de consulta ao FAQ institucional do Renovaí e à legislação externa (PNRS).'
    repositories = {
        'faq_chunks_repository': FaqChunksRepository,
        'normas_repository': FaqChunksRepository,
    }

    def __init__(self, retriever: FaqRetriever):
        self._retriever = retriever

    @property
    def retriever(self) -> FaqRetriever:
        return self._retriever

    @Toolkit.timed(categoria='tool', nome='faq_retriever')
    def faq_retriever(self, pergunta: str) -> dict:
        """Busca no FAQ oficial do Renovaí os trechos mais relevantes para responder dúvidas sobre a plataforma, perfis, regras e termos de uso."""
        Toolkit.log.debug('acessou tool - faq_retriever')

        if not pergunta or not pergunta.strip():
            return {'status': 'error', 'message': 'Pergunta não informada.'}

        resultados = self._retriever.buscar_faq(pergunta)

        if not resultados:
            return {'status': 'error', 'message': 'Nenhum trecho do FAQ relevante foi encontrado.'}

        return {
            'status': 'ok',
            'chunks': [
                {
                    'fonte': r.fonte,
                    'artigo': r.artigo,
                    'conteudo': r.conteudo,
                }
                for r in resultados
            ],
        }

    @Toolkit.timed(categoria='tool', nome='normas_retriever')
    def normas_retriever(self, pergunta: str) -> dict:
        """Busca na legislação ambiental e Política Nacional de Resíduos Sólidos (PNRS / Lei nº 12.305/2010) os artigos, definições, princípios, diretrizes e responsabilidades aplicáveis."""
        Toolkit.log.debug('acessou tool - normas_retriever')

        if not pergunta or not pergunta.strip():
            return {'status': 'error', 'message': 'Pergunta não informada.'}

        resultados = self._retriever.buscar_normas(pergunta)

        if not resultados:
            return {'status': 'error', 'message': 'Nenhum artigo ou trecho da PNRS relevante foi encontrado.'}

        return {
            'status': 'ok',
            'chunks': [
                {
                    'fonte': r.fonte,
                    'artigo': r.artigo,
                    'conteudo': r.conteudo,
                }
                for r in resultados
            ],
        }

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.faq_retriever,
                name='faq_retriever',
                description=self.faq_retriever.__doc__,
            ),
            StructuredTool.from_function(
                func=self.normas_retriever,
                name='normas_retriever',
                description=self.normas_retriever.__doc__,
            ),
        ]
