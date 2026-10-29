"""
Módulo de configurações globais da aplicação.

Gerencia as variáveis de ambiente e parâmetros de configuração do sistema
utilizando o Pydantic Settings, garantindo validação de tipos, valores padrão
e carregamento automático do arquivo '.env'.

Classes:
    Settings: Define o schema das variáveis de ambiente da aplicação.

"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import (
    ClassVar, Dict, Tuple, List, Any
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    QDRANT_API_KEY: str
    LANGSMITH_API_KEY: str | None = None

    JUDGE_ENGINE: str = 'deterministic'
    JUDGE_PRELOAD: bool = False

    DATABASE_URL: str
    QDRANT_URL: str
    REDIS_URL: str = ''
    MONGODB_URI: str

    NEO4J_URI: str = 'bolt://localhost:7687'
    NEO4J_USER: str = 'neo4j'
    NEO4J_PASSWORD: str = ''
    NEO4J_DATABASE: str = 'neo4j'

    # URLs opcionais para comunicação HTTP remota entre o Host Graph e os
    # Microsserviços A2A de Cooperativa (/a2a/cooperativa) e Empresa (/a2a/empresa).
    COOPERATIVA_A2A_URL: str = ''
    EMPRESA_A2A_URL: str = ''

    FAQ_PATH: str
    PNRS_PATH: str = str(BASE_DIR / 'data' / 'PNRS.txt')

    LANGSMITH_TRACING: bool = False
    LANGSMITH_PROJECT: str = 'renovai-ia'

    SESSAO_TTL_MIN: int = 120

    def model_post_init(self, __context: Any) -> None:
        import os
        os.environ['JUDGE_ENGINE'] = self.JUDGE_ENGINE
        if self.JUDGE_PRELOAD:
            os.environ['JUDGE_PRELOAD'] = 'true'
        if self.LANGSMITH_TRACING:
            os.environ['LANGSMITH_TRACING'] = 'true'
            if self.LANGSMITH_API_KEY:
                os.environ['LANGSMITH_API_KEY'] = self.LANGSMITH_API_KEY
            if self.LANGSMITH_PROJECT:
                os.environ['LANGSMITH_PROJECT'] = self.LANGSMITH_PROJECT

    MONGO_DB_NAME: str = 'mongo_dbrenovai'

    MONGO_COLLECTION_SESSAO:   str = 'sessao'
    QDRANT_COLLECTION_MEMORIA: str = 'memoria_resumos'
    QDRANT_COLLECTION_FAQ:     str = 'faq_chunks'
    QDRANT_COLLECTION_NORMAS:  str = 'normas_residuos'

    ROTAS_COOPERATIVA: Tuple[str, ...] = ('estoque', 'coleta', 'financeiro', 'logistica')
    ROTAS_EMPRESA: Tuple[str, ...] = ('pedidos',)

    AGENT_LLM_MAP: Dict[str, Tuple[str, str]] = {
        'router_agent':           ('GROQ',   'LOW'),
        'faq_agent':              ('GEMINI', 'MEDIUM'),
        'material_estoque_agent': ('GEMINI', 'HIGH'),
        'recomendacao_agent':     ('GEMINI', 'HIGH'),
        'orchestrator_agent':     ('GROQ',   'MEDIUM'),
        'resumo_agent':           ('GROQ',   'LOW'),
        'guardrail_entrada':      ('GROQ',   'LOW'),
        'guardrail_saida':        ('GROQ',   'LOW'),
        'coleta_agent':           ('GEMINI', 'HIGH'),
        'financeiro_agent':       ('GEMINI', 'HIGH'),
    }
    GUARDRAIL_ROTAS_REVISAO: Tuple[str, ...] = ('financeiro',)
    AGENT_REPOSITORY_MAP: Dict[str, List[str]] = {
        'material_estoque_agent': ['material_repository', 'estoque_repository', 'movimentacao_estoque_repository'],
        'coleta_agent':           ['coleta_repository', 'triagem_repository', 'funcionario_repository', 'equipe_repository', 'material_repository'],
        'financeiro_agent':       ['rateio_repository', 'rateio_funcionario_repository', 'despesa_repository', 'funcionario_repository'],
    }

    _CAMPOS_OBRIGATORIOS: ClassVar[tuple[str, ...]] = (
        'GEMINI_API_KEY',
        'GROQ_API_KEY',
        'DATABASE_URL',
    )

    def validar_config(self) -> list[str]:
        """Devolve a lista de problemas de configuração (vazia = tudo certo)."""
        problemas = []
        for nome in self._CAMPOS_OBRIGATORIOS:
            if not getattr(self, nome, None):
                problemas.append(f'Variável ausente no .env: {nome}')
        return problemas
