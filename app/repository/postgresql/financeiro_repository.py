from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID

from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.repository.base import Snapshot, SnapshotRepository

import logging
log = logging.getLogger(__name__)


class Rateio(BaseModel):
    rateio_id: UUID
    cooperativa_id: UUID
    mes_referencia: str
    tipo_rateio: str
    status: str
    valor_total_arrecadado: Decimal
    valor_despesas_operacionais: Decimal
    valor_liquido_distribuido: Decimal
    quantidade_cooperados: int
    data_abertura: datetime
    data_fechamento: Optional[datetime] = None


class RateioFuncionario(BaseModel):
    rateio_funcionario_id: UUID
    rateio_id: UUID
    cooperativa_id: UUID
    cooperado_id: UUID
    nome_cooperado: str
    mes_referencia: str
    porcentagem_individual: Decimal
    valor_recebido: Decimal
    horas_trabalhadas: Optional[Decimal] = None
    peso_triado_kg: Optional[Decimal] = None


class Despesa(BaseModel):
    despesa_id: UUID
    cooperativa_id: UUID
    categoria: str
    descricao: str
    valor: Decimal
    data_despesa: datetime
    mes_referencia: str
    status_pagamento: str


class FinanceiroSnapshot(Snapshot[Rateio]):
    itens_funcionarios: List[RateioFuncionario] = []
    despesas: List[Despesa] = []


QUERY_RATEIOS_POR_COOPERATIVA = """
    SELECT
        r.rateio_id,
        r.cooperativa_id,
        r.mes_referencia,
        r.tipo_rateio,
        r.status,
        r.valor_total_arrecadado,
        r.valor_despesas_operacionais,
        r.valor_liquido_distribuido,
        r.quantidade_cooperados,
        r.data_abertura,
        r.data_fechamento
    FROM rateios r
    WHERE r.cooperativa_id = %s
    ORDER BY r.data_abertura DESC;
"""

QUERY_RATEIOS_FUNCIONARIOS_POR_COOPERATIVA = """
    SELECT
        rf.rateio_funcionario_id,
        rf.rateio_id,
        r.cooperativa_id,
        rf.cooperado_id,
        rf.nome_cooperado,
        r.mes_referencia,
        rf.porcentagem_individual,
        rf.valor_recebido,
        rf.horas_trabalhadas,
        rf.peso_triado_kg
    FROM rateios_funcionarios rf
    INNER JOIN rateios r ON rf.rateio_id = r.rateio_id
    WHERE r.cooperativa_id = %s
    ORDER BY r.mes_referencia DESC, rf.nome_cooperado;
"""

QUERY_DESPESAS_POR_COOPERATIVA = """
    SELECT
        d.despesa_id,
        d.cooperativa_id,
        d.categoria,
        d.descricao,
        d.valor,
        d.data_despesa,
        d.mes_referencia,
        d.status_pagamento
    FROM despesas d
    WHERE d.cooperativa_id = %s
    ORDER BY d.data_despesa DESC;
"""


class FinanceiroRepository(SnapshotRepository[Rateio, ConnectionPool]):
    def __init__(self, db: ConnectionPool):
        super().__init__(db)

    def get_snapshot(self, cooperativa_id: UUID) -> FinanceiroSnapshot:
        log.debug('chegou no financeiro_repository e tirou snapshot para cooperativa %s', cooperativa_id)
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(QUERY_RATEIOS_POR_COOPERATIVA, [cooperativa_id])
                itens_rateio = [Rateio(**item) for item in cur.fetchall()]

                cur.execute(QUERY_RATEIOS_FUNCIONARIOS_POR_COOPERATIVA, [cooperativa_id])
                itens_func = [RateioFuncionario(**item) for item in cur.fetchall()]

                cur.execute(QUERY_DESPESAS_POR_COOPERATIVA, [cooperativa_id])
                itens_despesa = [Despesa(**item) for item in cur.fetchall()]

        return FinanceiroSnapshot(
            cooperativa_id=cooperativa_id,
            capturado_em=datetime.now(),
            itens=itens_rateio,
            itens_funcionarios=itens_func,
            despesas=itens_despesa,
        )
