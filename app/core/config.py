"""
Módulo de configurações globais da aplicação.

Gerencia as variáveis de ambiente e parâmetros de configuração do sistema
utilizando o Pydantic Settings, garantindo validação de tipos, valores padrão 
e carregamento automático do arquivo '.env'.

Classes:
    Settings: Define o schema das variáveis de ambiente da aplicação.

"""
from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import (
    ClassVar, Dict, Tuple, List
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
    )

    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    MONGODB_URI: str
    DATABASE_URL: str

    AGENT_LLM_MAP: Dict[str, Tuple[str, str]] = {
        'router_agent':           ('GROQ',   'LOW'),
        'material_estoque_agent': ('GEMINI', 'HIGH'),
        'orchestrator_agent':     ('GROQ',   'LOW'),
        'resumo_agent':           ('GROQ',   'LOW'),
    }
    AGENT_REPOSITORY_MAP: Dict[str, List[str]] = {
        'material_estoque_agent': ['material_repository','estoque_repository', 'movimentacao_estoque_repository']
    }
    ROUTE_NODE_MAP: Dict[str, str] = {
        'material_estoque': 'material_estoque_agent',
    }

    _CAMPOS_OBRIGATORIOS: ClassVar[tuple[str, ...]] = (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "DATABASE_URL",
    )

    def validar_config(self) -> list[str]:
        """Devolve a lista de problemas de configuração (vazia = tudo certo)."""
        problemas = []
        for nome in self._CAMPOS_OBRIGATORIOS:
            if not getattr(self, nome, None):
                problemas.append(f"Variável ausente no .env: {nome}")
        return problemas 