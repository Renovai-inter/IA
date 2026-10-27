import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig, RunnableLambda

from app.repository.base import Repository
from app.agents.base import BaseAgent

from app.graph.state import EspecialistaOutput, GraphState, resolver_contexto_dominio

if TYPE_CHECKING:
    from app.a2a.client import A2AClient

# Pool compartilhado: os snapshots são I/O de banco independentes entre si.
_SNAPSHOT_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix='snapshot')

MENSAGEM_RESTRICAO_COOPERATIVA = (
    'Essa consulta está disponível apenas para perfis de cooperativa. '
    'Se você representa uma empresa, posso ajudar com recomendações de '
    'cooperativas parceiras para compra de recicláveis ou dúvidas sobre o Renovaí.'
)

MENSAGEM_RESTRICAO_EMPRESA = (
    'Essa consulta de recomendação de fornecedores está disponível apenas para perfis de empresa compradora. '
    'Se você representa uma cooperativa, posso ajudar com estoque, coletas, triagens, rateio financeiro, '
    'logística de rotas ou dúvidas sobre o Renovaí.'
)


def _enriquecer_config(
    config: RunnableConfig | None,
    cooperativa_id,
    empresa_id,
) -> RunnableConfig:
    base_cfg = dict(config or {})
    configurable = dict(base_cfg.get('configurable') or {})
    if cooperativa_id is not None:
        configurable['cooperativa_id'] = str(cooperativa_id)
    if empresa_id is not None:
        configurable['empresa_id'] = str(empresa_id)
    base_cfg['configurable'] = configurable
    return base_cfg


def make_repo_backed_runners(
    agent: BaseAgent,
    repos: dict[str, Repository],
    exigir_empresa: bool = False,
):
    def sync_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
        cooperativa_id, empresa_id, _ = resolver_contexto_dominio(state, config)

        if exigir_empresa and empresa_id is None and cooperativa_id is not None:
            return {
                'messages': [{'role': 'assistant', 'content': MENSAGEM_RESTRICAO_EMPRESA}],
                'agentes_chamados': [agent.nome_agente],
                'especialista_outputs': {
                    agent.nome_agente: EspecialistaOutput(
                        conteudo=MENSAGEM_RESTRICAO_EMPRESA,
                        fonte=agent.nome_agente,
                    )
                },
            }

        if repos and cooperativa_id is None:
            return {
                'messages': [{'role': 'assistant', 'content': MENSAGEM_RESTRICAO_COOPERATIVA}],
                'agentes_chamados': [agent.nome_agente],
                'especialista_outputs': {
                    agent.nome_agente: EspecialistaOutput(
                        conteudo=MENSAGEM_RESTRICAO_COOPERATIVA,
                        fonte=agent.nome_agente,
                    )
                },
            }

        futuros = {
            nome.replace('repository', 'snapshot'): _SNAPSHOT_POOL.submit(repo.get_snapshot, cooperativa_id)
            for nome, repo in repos.items()
        }
        snapshots = {nome: futuro.result() for nome, futuro in futuros.items()}
        novo_state = {
            **state,
            'cooperativa_id': cooperativa_id,
            'empresa_id': empresa_id,
            'snapshots': snapshots,
        }
        return agent.run(
            state=novo_state,
            config=_enriquecer_config(config, cooperativa_id, empresa_id),
        )

    async def async_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
        cooperativa_id, empresa_id, _ = resolver_contexto_dominio(state, config)

        if exigir_empresa and empresa_id is None and cooperativa_id is not None:
            return {
                'messages': [{'role': 'assistant', 'content': MENSAGEM_RESTRICAO_EMPRESA}],
                'agentes_chamados': [agent.nome_agente],
                'especialista_outputs': {
                    agent.nome_agente: EspecialistaOutput(
                        conteudo=MENSAGEM_RESTRICAO_EMPRESA,
                        fonte=agent.nome_agente,
                    )
                },
            }

        if repos and cooperativa_id is None:
            return {
                'messages': [{'role': 'assistant', 'content': MENSAGEM_RESTRICAO_COOPERATIVA}],
                'agentes_chamados': [agent.nome_agente],
                'especialista_outputs': {
                    agent.nome_agente: EspecialistaOutput(
                        conteudo=MENSAGEM_RESTRICAO_COOPERATIVA,
                        fonte=agent.nome_agente,
                    )
                },
            }

        if repos:
            tarefas = [
                asyncio.to_thread(repo.get_snapshot, cooperativa_id)
                for repo in repos.values()
            ]
            resultados = await asyncio.gather(*tarefas)
            snapshots = {
                nome.replace('repository', 'snapshot'): res
                for nome, res in zip(repos.keys(), resultados)
            }
        else:
            snapshots = {}

        novo_state = {
            **state,
            'cooperativa_id': cooperativa_id,
            'empresa_id': empresa_id,
            'snapshots': snapshots,
        }
        cfg_enriquecido = _enriquecer_config(config, cooperativa_id, empresa_id)

        if hasattr(agent, "arun"):
            return await agent.arun(state=novo_state, config=cfg_enriquecido)
        return agent.run(state=novo_state, config=cfg_enriquecido)

    return sync_node, async_node


def make_repo_backed_node(
    agent: BaseAgent,
    repos: dict[str, Repository],
    exigir_empresa: bool = False,
):
    sync_node, async_node = make_repo_backed_runners(agent, repos, exigir_empresa=exigir_empresa)
    return RunnableLambda(sync_node, afunc=async_node)


def make_a2a_node(rota: str, a2a_client: 'A2AClient'):
    """Cria um nó no Grafo Principal (Host Graph) que encaminha a execução da rota
    especialista via protocolo Google A2A (`tasks/send`) para o Microsserviço de
    Cooperativa (`/a2a/cooperativa`) ou Microsserviço de Empresa (`/a2a/empresa`)."""

    def sync_a2a_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
        cooperativa_id, empresa_id, _ = resolver_contexto_dominio(state, config)
        cfg = ((config or {}).get('configurable') or {}) if isinstance(config, dict) else {}

        rpc_resp = a2a_client.send_task(
            skill=rota,
            cooperativa_id=str(cooperativa_id) if cooperativa_id else None,
            empresa_id=str(empresa_id) if empresa_id else None,
            messages=list(state.get('messages') or []),
            session_id=cfg.get('thread_id'),
            user_id=cfg.get('user_id'),
        )
        return a2a_client.extrair_estado_grafo(rpc_resp)

    async def async_a2a_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
        cooperativa_id, empresa_id, _ = resolver_contexto_dominio(state, config)
        cfg = ((config or {}).get('configurable') or {}) if isinstance(config, dict) else {}

        rpc_resp = await a2a_client.asend_task(
            skill=rota,
            cooperativa_id=str(cooperativa_id) if cooperativa_id else None,
            empresa_id=str(empresa_id) if empresa_id else None,
            messages=list(state.get('messages') or []),
            session_id=cfg.get('thread_id'),
            user_id=cfg.get('user_id'),
        )
        return a2a_client.extrair_estado_grafo(rpc_resp)

    return RunnableLambda(sync_a2a_node, afunc=async_a2a_node)
