import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
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
    _DEFAULT_SNAPSHOT_TTL: float = 60.0

    def __init__(self, db: Tdb):
        super().__init__(db)
        self._snapshot_cache: Dict[UUID, Tuple[float, Snapshot[T]]] = {}

    @abstractmethod
    def get_snapshot(self, cooperativa_id: UUID) -> Snapshot[T]:
        raise NotImplementedError

    def get_snapshot_cached(self, cooperativa_id: UUID, ttl: Optional[float] = None) -> Snapshot[T]:
        val_ttl = ttl if ttl is not None else self._DEFAULT_SNAPSHOT_TTL
        agora = time.monotonic()
        hit = self._snapshot_cache.get(cooperativa_id)
        if hit and (agora - hit[0]) < val_ttl:
            return hit[1]
        snap = self.get_snapshot(cooperativa_id)
        self._snapshot_cache[cooperativa_id] = (agora, snap)
        return snap
