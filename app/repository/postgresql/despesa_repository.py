from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID
import logging

from psycopg_pool import ConnectionPool
from pydantic import BaseModel, field_validator

from app.repository.base import Snapshot, SnapshotRepository
from app.repository.postgresql.rateio_funcionario_repository import _normalizar_mes_referencia

log = logging.getLogger(__name__)


class Despesa(BaseModel):
    lancamento_id: UUID
    despesa_id: UUID
    cooperativa_id: UUID
    nome_despesa: str
    tipo_despesa: str
    esta_ativa: bool = True
    valor: Decimal
    mes_referencia: date
    data_lancamento: datetime

    @field_validator('mes_referencia', mode='before')
    @classmethod
    def _validar_mes(cls, v: Any) -> date:
        return _normalizar_mes_referencia(v)


class DespesaSnapshot(Snapshot[Despesa]):
    pass


QUERY_DESPESAS_POR_COOPERATIVA = """
    SELECT
        ld.lancamento_id,
        d.despesa_id,
        d.cooperativa_id,
        d.nome AS nome_despesa,
        d.tipo_despesa,
        d.esta_ativa,
        ld.valor,
        ld.mes_referencia,
        ld.data_lancamento
    FROM lancamentos_despesas ld
    INNER JOIN despesas d
        ON ld.despesa_id = d.despesa_id
    WHERE d.cooperativa_id = %s
    ORDER BY ld.mes_referencia DESC, ld.data_lancamento DESC;
"""


class DespesaRepository(SnapshotRepository[Despesa, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> DespesaSnapshot:
        log.debug('chegou no despesa_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_DESPESAS_POR_COOPERATIVA, [cooperativa_id])
                itens = [Despesa(**item) for item in cur.fetchall()]

        return DespesaSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )
