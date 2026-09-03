from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """O que o navegador envia no POST /chat."""
    session_id: str = Field(..., examples=["usuario_id"])
    perfil_id:  str = Field(..., examples=["empresa_id", "cooperativa_id"])
    pergunta:   str = Field(..., min_length=1, examples=["Quantas triagens foram realizadas hoje?"])