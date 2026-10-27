from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """Payload enviado no POST /chat com roteamento direto por domínio (Cooperativa ou Empresa)."""
    session_id:     str = Field(..., examples=['sessao_id'])
    cooperativa_id: Optional[str] = Field(default=None, examples=['550e8400-e29b-41d4-a716-446655440000'])
    empresa_id:     Optional[str] = Field(default=None, examples=['660e8400-e29b-41d4-a716-446655440001'])
    user_id:        Optional[str] = Field(default=None, examples=['usuario_id'])
    pergunta:       str = Field(..., min_length=1, examples=['Quantas triagens foram realizadas hoje?'])

    @model_validator(mode='after')
    def _validar_identificador_dominio(self) -> 'ChatRequest':
        coop = (self.cooperativa_id or '').strip() or None
        emp = (self.empresa_id or '').strip() or None
        self.cooperativa_id = coop
        self.empresa_id = emp

        if bool(coop) == bool(emp):
            raise ValueError(
                'Informe exatamente um identificador de domínio: cooperativa_id (fluxo Cooperativa) '
                'ou empresa_id (fluxo Empresa).'
            )
        return self

    @property
    def tipo_fluxo(self) -> Literal['COOPERATIVA', 'EMPRESA']:
        return 'COOPERATIVA' if self.cooperativa_id else 'EMPRESA'

    @property
    def entidade_id(self) -> str:
        return (self.cooperativa_id or self.empresa_id or '')
