from abc import ABC
from typing import Generic, TypeVar
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

from psycopg_pool import ConnectionPool

T = TypeVar('T')

class Snapshot(BaseModel, Generic[T]):
    cooperativa_id: UUID
    capturado_em: datetime
    itens: list[T]

class Repository(ABC, Generic[T]):
    def __init__(self, db: ConnectionPool):
        self._db = db