from abc import ABC
from uuid import UUID
from datetime import datetime

from psycopg_pool import ConnectionPool

class Repository[T](ABC):
    def __init__(self, db: ConnectionPool):
        self._db = db

class Snapshot[T](ABC):
    cooperativa_id: UUID
    capturado_em: datetime
    itens: list[T]