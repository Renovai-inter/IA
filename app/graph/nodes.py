from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableConfig

from app.repository.base import Repository
from app.agents.base import BaseAgent

from app.graph.state import EspecialistaOutput, GraphState

# Pool compartilhado: os snapshots são I/O de banco independentes entre si.
_SNAPSHOT_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix='snapshot')


def make_repo_backed_node(agent: BaseAgent, repos: dict[str, Repository]):
    def node(state: GraphState, config: RunnableConfig) -> dict:
        perfil_ctx = state.get('perfil_ctx')
        cooperativa_id = perfil_ctx.cooperativa_id if perfil_ctx else None

        if repos and cooperativa_id is None:
            texto = (
                'Essa consulta está disponível apenas para perfis de cooperativa. '
                'Se você representa uma empresa, posso ajudar com dúvidas sobre '
                'como o Renovaí funciona.'
            )
            return {
                'messages'            : [{'role': 'assistant', 'content': texto}],
                'agentes_chamados'    : [agent.nome_agente],
                'especialista_outputs': {
                    agent.nome_agente: EspecialistaOutput(conteudo=texto, fonte=agent.nome_agente)
                },
            }

        futuros = {
            nome.replace('repository', 'snapshot'): _SNAPSHOT_POOL.submit(repo.get_snapshot, cooperativa_id)
            for nome, repo in repos.items()
        }
        snapshots = {nome: futuro.result() for nome, futuro in futuros.items()}
        return agent.run(
            state={**state, 'snapshots': snapshots},
            config=config
        )
    return node