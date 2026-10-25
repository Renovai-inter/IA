from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID
import logging

from psycopg_pool import ConnectionPool
from pydantic import BaseModel, field_validator

from app.repository.base import Snapshot, SnapshotRepository

log = logging.getLogger(__name__)


def _normalizar_mes_referencia(v: Any) -> date:
    if isinstance(v, datetime):
        return date(v.year, v.month, 1)
    if isinstance(v, date):
        return date(v.year, v.month, 1)
    if isinstance(v, str):
        texto = v.strip()
        if len(texto) == 7 and texto[4] == '-':
            ano, mes = texto.split('-')
            return date(int(ano), int(mes), 1)
        dt = date.fromisoformat(texto[:10])
        return date(dt.year, dt.month, 1)
    raise ValueError(f'Formato inválido para mes_referencia: {v!r}')


class RateioFuncionario(BaseModel):
    rateio_funcionario_id: UUID
    rateio_id: UUID
    cooperativa_id: UUID
    cooperado_id: UUID
    nome_cooperado: str
    mes_referencia: date
    tipo_rateio: Optional[str] = None
    valor_rateio: Decimal
    data_rateio: datetime

    @field_validator('mes_referencia', mode='before')
    @classmethod
    def _validar_mes(cls, v: Any) -> date:
        return _normalizar_mes_referencia(v)


class RateioFuncionarioSnapshot(Snapshot[RateioFuncionario]):
    pass


QUERY_RATEIOS_FUNCIONARIOS_POR_COOPERATIVA = """
    SELECT
        rf.rateio_funcionario_id,
        rf.rateio_id,
        r.cooperativa_id,
        rf.cooperado_id,
        u.nome AS nome_cooperado,
        r.mes_referencia,
        tr.tipo_rateio,
        rf.valor_rateio,
        r.data_rateio
    FROM rateios_funcionarios rf
    INNER JOIN rateios r
        ON rf.rateio_id = r.rateio_id
    INNER JOIN funcionarios f
        ON rf.cooperado_id = f.funcionario_id
    INNER JOIN usuarios u
        ON f.usuario_id = u.usuario_id
    LEFT JOIN tipos_rateios tr
        ON r.tipo_rateio_id = tr.tipo_rateio_id
    WHERE r.cooperativa_id = %s
    ORDER BY r.mes_referencia DESC, u.nome;
"""


class RateioFuncionarioRepository(SnapshotRepository[RateioFuncionario, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> RateioFuncionarioSnapshot:
        log.debug('chegou no rateio_funcionario_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_RATEIOS_FUNCIONARIOS_POR_COOPERATIVA, [cooperativa_id])
                itens = [RateioFuncionario(**item) for item in cur.fetchall()]

        return RateioFuncionarioSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )
