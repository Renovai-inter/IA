from abc import ABC, abstractmethod
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

from psycopg_pool import ConnectionPool
from pymongo import MongoClient


class Snapshot[T](BaseModel):
    cooperativa_id: UUID
    capturado_em: datetime
    itens: list[T]

class Repository[T](ABC):
    def __init__(self, db: ConnectionPool | MongoClient):
        self._db = db

class SnapshotRepository[T](Repository[T]):
    def __init__(self, db):
        super().__init__(db)

    @abstractmethod
    def get_snapshot(self, cooperativa_id: UUID) -> Snapshot[T]:
        raise NotImplementedError