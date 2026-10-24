from datetime import datetime
from typing import Optional
from uuid import UUID
import logging

from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.repository.base import Snapshot, SnapshotRepository

log = logging.getLogger(__name__)


class Funcionario(BaseModel):
    funcionario_id: UUID
    usuario_id: UUID
    cooperativa_id: UUID
    nome: str
    email: str
    cpf: str
    cargo_id: UUID
    cargo: str
    status_funcionario: str
    esta_ativo: bool
    esta_afastado: bool
    data_admissao: datetime
    data_desligamento: Optional[datetime] = None


class FuncionarioSnapshot(Snapshot[Funcionario]):
    pass


QUERY_FUNCIONARIOS_POR_COOPERATIVA = """
    SELECT
        f.funcionario_id,
        f.usuario_id,
        f.cooperativa_id,
        u.nome,
        u.email,
        u.cpf,
        f.cargo_id,
        ca.cargo,
        f.status_funcionario,
        f.esta_ativo,
        f.esta_afastado,
        f.data_admissao,
        f.data_desligamento
    FROM funcionarios f
    INNER JOIN usuarios u
        ON f.usuario_id = u.usuario_id
    INNER JOIN cargos ca
        ON f.cargo_id = ca.cargo_id
    WHERE f.cooperativa_id = %s
    ORDER BY u.nome;
"""


class FuncionarioRepository(SnapshotRepository[Funcionario, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> FuncionarioSnapshot:
        log.debug('chegou no funcionario_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_FUNCIONARIOS_POR_COOPERATIVA, [cooperativa_id])
                itens = [Funcionario(**item) for item in cur.fetchall()]

        return FuncionarioSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )
