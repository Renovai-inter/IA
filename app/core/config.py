"""
Módulo de configurações globais da aplicação.

Gerencia as variáveis de ambiente e parâmetros de configuração do sistema
utilizando o Pydantic Settings, garantindo validação de tipos, valores padrão 
e carregamento automático do arquivo '.env'.

Classes:
    Settings: Define o schema das variáveis de ambiente da aplicação.

Funções:
    get_settings: Retorna uma instância singleton da classe Settings
        memorizada via 'lru_cache'.
"""
from pydantic_settings import BaseSettings
from typing import (
    Dict, Tuple
)

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    MONGO_URI: str
    DATABASE_URL: str

    AGENT_LLM_MAP: Dict[str, Tuple[str, str]] = {
        'router_agent': ('GROQ', 'LOW'),
    }

    class Config:
        env_file = '.env'