from langchain_core.tools import StructuredTool

from app.rag.retriever import FaqRetriever
from app.repository.qdrant.faq_chunks_repository import FaqChunksRepository
from app.tools.base import Toolkit


class FaqChunksToolkit(Toolkit):
    nome = 'memoria_toolkit'
    descricao = 'Tools de consulta de conversas anteriores do usuário.'
    repositories = {
        'faq_chunks_repository': FaqChunksRepository,
    }

    def __init__(self, retriever: FaqRetriever):
        self.retriever = retriever

    def faq_retriever(self, pergunta: str):
        """Busca no FAQ oficial do Renovaí os trechos mais relevantes para responder a pergunta."""
        resultados = self.retriever.buscar(pergunta)

        if pergunta in (None, ''):
            return {'status': 'error', 'message': 'Pergunta não relacionada ao faq.'}

        retriever_output = {
            'status': 'ok',
            'chunks': len(resultados)
        }

        return retriever_output | {
            r.pagina: r.conteudo
            for r in resultados
        }

    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.faq_retriever,
                name='faq_retriever',
                description=self.faq_retriever.__doc__,
            ),
        ]