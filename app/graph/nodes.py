from langchain_core.runnables import RunnableConfig

from app.repository.base import Repository
from app.agents.base import BaseAgent

from app.graph.state import GraphState

def make_repo_backed_node(agent: BaseAgent, repos: dict[str, Repository]):
    def node(state: GraphState, config: RunnableConfig) -> dict:
        cooperativa_id = state['perfil_ctx'].cooperativa_id
        snapshots = {nome.replace('repository', 'snapshot'): repo.get_snapshot(cooperativa_id) for nome, repo in repos.items()}
        return agent.run(
            state={**state, 'snapshots': snapshots},
            config=config
        )
    return node