from app.repository.base import Repository
from psycopg_pool import ConnectionPool

from pydantic import BaseModel
from typing import Literal, Optional
from uuid import UUID
import time

class PerfilContext(BaseModel):
    perfil_id: UUID
    tipo: Literal['EMPRESA', 'COOPERATIVA']
    empresa_id: Optional[UUID]
    cooperativa_id: Optional[UUID]


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
                tipo = 'EMPRESA' if row['empresa_id'] else 'COOPERATIVA'
                
                return PerfilContext(
                    perfil_id=perfil_id,
                    tipo=tipo,
                    empresa_id=row['empresa_id'],
                    cooperativa_id=row['cooperativa_id'],
                )
