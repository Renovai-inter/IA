"""Pacote raiz da aplicação Renovai IA."""
# Garante que as extensões do checkpointer (serialização de modelos Pydantic)
# sejam aplicadas quando qualquer módulo da aplicação for importado.
import app.core.checkpointer  # noqa: F401
