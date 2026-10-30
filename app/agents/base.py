from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from app.graph.state import GraphState

from typing import List
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    # Quantas mensagens recentes do thread cada agente enxerga. O MemorySaver
    # acumula o thread inteiro; sem janela, o custo de tokens cresce a cada turno.
    # 6 cobre o turno atual (pergunta + handoff do roteador + JSON do especialista)
    # mais um turno anterior de contexto.
    janela_mensagens: int = 6

    def __init__(self, llm: BaseChatModel, system_prompt: str, tools: List[StructuredTool] | None = None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []

    def _mensagens_recentes(self, state: GraphState) -> list:
        return list(state['messages'])[-self.janela_mensagens:]

    def _obter_texto_mensagem(self, msg) -> str:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        return "".join(
            bloco.get("text", "")
            for bloco in content
            if isinstance(bloco, dict) and bloco.get("type") == "text"
    )
    
    def _config_interno(self, config: RunnableConfig | None, **extras) -> RunnableConfig:
        """Monta o config do agente interno (create_agent) a partir do config do nó.

        Repassa só as chaves de negócio de `configurable` (thread_id, user_id,
        perfil_id, ...) — as chaves internas do LangGraph (`__pregel_*`,
        `checkpoint_*`) ficam de fora para o agente interno não herdar o
        checkpointer do grafo pai. `extras` entra em `configurable` (ex.:
        snapshots), que é onde as tools leem com `config['configurable'][...]`.
        """
        configurable = (config or {}).get('configurable', {}) or {}
        filtrado = {
            k: v for k, v in configurable.items()
            if not k.startswith('__') and not k.startswith('checkpoint_')
        }
        novo_config = {'configurable': {**filtrado, **extras}}
        if 'callbacks' in config:
            novo_config['callbacks'] = config['callbacks']
        return novo_config

    @abstractmethod
    def run(self, state: GraphState, config: RunnableConfig):
        raise NotImplementedError

    @abstractmethod
    async def arun(self, state: GraphState, config: RunnableConfig):
        raise NotImplementedError