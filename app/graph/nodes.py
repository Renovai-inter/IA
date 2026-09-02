from app.repository.base import Repository
from app.agents.material_estoque_agent import MaterialEstoqueAgent

from app.graph.state import GraphState

def material_estoque_node(state: GraphState, agent: MaterialEstoqueAgent, repositories: list[Repository]) -> dict:
    cooperativa_id = state["perfil_ctx"].cooperativa_id
    snapshots = [repo.get_snapshot(cooperativa_id) for repo in repositories]

    enriched_state = {**state, "snapshots": snapshots}
    return agent.run(enriched_state)