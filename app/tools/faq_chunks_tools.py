from langchain_core.tools import StructuredTool

from app.rag.retriever import FaqRetriever
from app.repository.qdrant.faq_chunks_repository import FaqChunksRepository
from app.tools.base import Toolkit


class FaqToolkit(Toolkit):
    nome = 'faq_toolkit'
    descricao = 'Tools de consulta ao FAQ institucional do Renovaí.'
    repositories = {
        'faq_chunks_repository': FaqChunksRepository,
    }

    def __init__(self, retriever: FaqRetriever):
        self.retriever = retriever

    @Toolkit.timed(categoria='tool', nome='faq_retriever')
    def faq_retriever(self, pergunta: str) -> dict:
        """Busca no FAQ oficial do Renovaí os trechos mais relevantes para responder a pergunta."""
        Toolkit.log.debug('acessou tool - faq_retriever')

        if pergunta in (None, ''):
            return {'status': 'error', 'message': 'Pergunta não relacionada ao faq.'}

        resultados = self.retriever.buscar(pergunta)

        if not resultados:
            return {'status': 'error', 'message': 'Nenhum trecho do FAQ relevante foi encontrado.'}

        return {
            'status': 'ok',
            'chunks': [
                {'fonte': r.fonte, 'conteudo': r.conteudo}
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
        ]