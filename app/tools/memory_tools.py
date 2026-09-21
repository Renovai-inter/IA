from typing import Annotated

from langchain.tools import InjectedState
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from app.repository.mongodb.sessao_repository import SessaoRepository
from app.memory.mongo_memory import MongoMemory
from app.tools.base import Toolkit


class MemoriaToolkit(Toolkit):
    nome = 'memoria_toolkit'
    descricao = 'Tools de consulta de conversas anteriores do usuário.'
    repositories = {
        'sessao_repository': SessaoRepository
    }

    def __init__(self, mongo_memory: MongoMemory):
        self.mongo_memory = mongo_memory

    def buscar_historico(self, config: RunnableConfig) -> dict:
        """Consulta conversas ANTERIORES do usuário (sessões já encerradas).

        Use SOMENTE quando a resposta depende de algo dito numa conversa passada
        — preferências, decisões ou planos que o usuário mencionou antes.
        NÃO use para dados que estão no banco : para isso
        já existem as tools de consulta específicas"""
        config = (config or {}).get('configurable', {})
        user_id = config.get('user_id') or config.get('thread_id')
        perfil_id = config.get('perfil_id', None)

        if not user_id and not perfil_id:
            return {'status': 'error', 'message': 'Não foi possível identificar o usuário para buscar o histórico.'}

        historico = self.mongo_memory.recuperar_historico(user_id, perfil_id)

        if not historico:
            return {'status': 'error', 'message': 'Nenhuma conversa anterior relevante encontrada.'}

        historico_output = {
            'status' : 'ok',
            'sessoes': len(historico)
        }
        return historico_output | {
            h['data_inicio'].strftime('%d/%m/%Y') if hasattr(h['data_inicio'], "strftime") else str(h['data_inicio'])[:10] : h['resumo']
            for h in historico
        }


    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.buscar_historico,
                name='buscar_historico',
                description=self.buscar_historico.__doc__,
            ),
        ]