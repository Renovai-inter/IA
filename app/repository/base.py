from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class ConnectionFactory[T](ABC):
    def __init__(self, dsn: str):
       self.dsn = dsn

    @abstractmethod
    def connect(self, dsn: str) -> T:
        raise NotImplementedError
    @abstractmethod
    def close(self, conn: T) -> None:
        raise NotImplementedError

class Snapshot[T](BaseModel):
    cooperativa_id: UUID
    capturado_em: datetime
    itens: list[T]

class Repository[T, Tdb](ABC):
    def __init__(self, db: Tdb):
        self._db = db

class SnapshotRepository[T, Tdb](Repository[T, Tdb]):
    def __init__(self, db: Tdb):
        super().__init__(db)

    @abstractmethod
    def get_snapshot(self, cooperativa_id: UUID) -> Snapshot[T]:
        raise NotImplementedError