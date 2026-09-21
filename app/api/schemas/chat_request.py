from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """O que o navegador envia no POST /chat."""
    session_id: str = Field(..., examples=['sessao_id'])
    perfil_id:  str = Field(..., examples=['empresa_id', 'cooperativa_id'])
    user_id:    str = Field(default=None, examples=['usuario_id'])
    pergunta:   str = Field(..., min_length=1, examples=['Quantas triagens foram realizadas hoje?'])