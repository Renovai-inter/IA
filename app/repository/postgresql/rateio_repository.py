from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID
import logging

from psycopg_pool import ConnectionPool
from pydantic import BaseModel, field_validator

from app.repository.base import Snapshot, SnapshotRepository
from app.repository.postgresql.rateio_funcionario_repository import _normalizar_mes_referencia

log = logging.getLogger(__name__)


class Rateio(BaseModel):
    rateio_id: UUID
    cooperativa_id: UUID
    gestor_id: UUID
    nome_gestor: Optional[str] = None
    tipo_rateio_id: Optional[UUID] = None
    tipo_rateio: Optional[str] = None
    descricao_tipo_rateio: Optional[str] = None
    mes_referencia: date
    data_rateio: datetime
    valor_total_rateio: Decimal = Decimal('0')
    quantidade_cooperados: int = 0

    @field_validator('mes_referencia', mode='before')
    @classmethod
    def _validar_mes(cls, v: Any) -> date:
        return _normalizar_mes_referencia(v)


class RateioSnapshot(Snapshot[Rateio]):
    pass


QUERY_RATEIOS_POR_COOPERATIVA = """
    SELECT
        r.rateio_id,
        r.cooperativa_id,
        r.gestor_id,
        ug.nome AS nome_gestor,
        r.tipo_rateio_id,
        tr.tipo_rateio,
        tr.descricao AS descricao_tipo_rateio,
        r.mes_referencia,
        r.data_rateio,
        COALESCE(SUM(rf.valor_rateio), 0) AS valor_total_rateio,
        COUNT(rf.rateio_funcionario_id) AS quantidade_cooperados
    FROM rateios r
    LEFT JOIN tipos_rateios tr
        ON r.tipo_rateio_id = tr.tipo_rateio_id
    LEFT JOIN funcionarios g
        ON r.gestor_id = g.funcionario_id
    LEFT JOIN usuarios ug
        ON g.usuario_id = ug.usuario_id
    LEFT JOIN rateios_funcionarios rf
        ON r.rateio_id = rf.rateio_id
    WHERE r.cooperativa_id = %s
    GROUP BY
        r.rateio_id,
        r.cooperativa_id,
        r.gestor_id,
        ug.nome,
        r.tipo_rateio_id,
        tr.tipo_rateio,
        tr.descricao,
        r.mes_referencia,
        r.data_rateio
    ORDER BY r.mes_referencia DESC, r.data_rateio DESC;
"""


class RateioRepository(SnapshotRepository[Rateio, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> RateioSnapshot:
        log.debug('chegou no rateio_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_RATEIOS_POR_COOPERATIVA, [cooperativa_id])
                itens = [Rateio(**item) for item in cur.fetchall()]

        return RateioSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens,
        )
