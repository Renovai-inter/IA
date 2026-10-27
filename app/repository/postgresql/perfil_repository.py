from app.repository.base import Repository
from psycopg_pool import ConnectionPool

from pydantic import BaseModel
from typing import Literal, Optional, Union
from uuid import UUID, NAMESPACE_DNS, uuid5
import time


def _to_uuid(valor: Union[str, UUID, None]) -> Optional[UUID]:
    if valor is None:
        return None
    if isinstance(valor, UUID):
        return valor
    valor_str = str(valor).strip()
    if not valor_str:
        return None
    try:
        return UUID(valor_str)
    except ValueError:
        return uuid5(NAMESPACE_DNS, valor_str)


class PerfilContext(BaseModel):
    perfil_id: Optional[UUID] = None
    tipo: Literal['EMPRESA', 'COOPERATIVA']
    empresa_id: Optional[UUID] = None
    cooperativa_id: Optional[UUID] = None

    @classmethod
    def from_request(
        cls,
        cooperativa_id: Union[str, UUID, None] = None,
        empresa_id: Union[str, UUID, None] = None,
    ) -> 'PerfilContext':
        coop_uuid = _to_uuid(cooperativa_id)
        emp_uuid = _to_uuid(empresa_id)
        if coop_uuid is not None and emp_uuid is not None:
            raise ValueError('Informe apenas cooperativa_id ou empresa_id, não ambos.')
        if coop_uuid is None and emp_uuid is None:
            raise ValueError('É obrigatório informar cooperativa_id ou empresa_id.')

        if coop_uuid is not None:
            return cls(
                perfil_id=coop_uuid,
                tipo='COOPERATIVA',
                empresa_id=None,
                cooperativa_id=coop_uuid,
            )
        return cls(
            perfil_id=emp_uuid,
            tipo='EMPRESA',
            empresa_id=emp_uuid,
            cooperativa_id=None,
        )

    @property
    def identificador(self) -> str:
        alvo = self.cooperativa_id or self.empresa_id or self.perfil_id
        return str(alvo) if alvo is not None else ''


class PerfilRepository(Repository[PerfilContext, ConnectionPool]):
    _TTL_SEGUNDOS = 300  # perfil muda raramente; evita 1 query por turno

    def __init__(self, db: ConnectionPool):
        super().__init__(db)
        self._cache: dict[UUID, tuple[float, PerfilContext]] = {}

    def resolve_perfil(self, perfil_id: UUID) -> PerfilContext:
        hit = self._cache.get(perfil_id)
        if hit and time.monotonic() - hit[0] < self._TTL_SEGUNDOS:
            return hit[1]
        ctx = self._resolve_perfil_db(perfil_id)
        self._cache[perfil_id] = (time.monotonic(), ctx)
        return ctx

    def _resolve_perfil_db(self, perfil_id: UUID) -> PerfilContext:
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                query = 'SELECT empresa_id, cooperativa_id FROM perfis WHERE perfil_id = %s'
                params = [perfil_id]

                cur.execute(query, params)
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Perfil {perfil_id} não encontrado no banco de dados.")
                tipo = 'EMPRESA' if row['empresa_id'] else 'COOPERATIVA'
                
                return PerfilContext(
                    perfil_id=perfil_id,
                    tipo=tipo,
                    empresa_id=row['empresa_id'],
                    cooperativa_id=row['cooperativa_id'],
                )
