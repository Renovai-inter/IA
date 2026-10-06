from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableConfig

from app.repository.base import Repository
from app.agents.base import BaseAgent

from app.graph.state import GraphState

# Pool compartilhado: os snapshots são I/O de banco independentes entre si.
_SNAPSHOT_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix='snapshot')


def make_repo_backed_node(agent: BaseAgent, repos: dict[str, Repository]):
    def node(state: GraphState, config: RunnableConfig) -> dict:
        cooperativa_id = state['perfil_ctx'].cooperativa_id
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