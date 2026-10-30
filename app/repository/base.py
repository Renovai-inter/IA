from abc import ABC, abstractmethod
from datetime import datetime
import threading
import time
from typing import Dict, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel


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
    _MAX_CACHE_ENTRIES: int = 32

    def __init__(self, db: Tdb):
        super().__init__(db)
        self._snapshot_cache: Dict[UUID, Tuple[float, Snapshot[T]]] = {}
        self._lock = threading.Lock()

    @abstractmethod
    def get_snapshot(self, cooperativa_id: UUID) -> Snapshot[T]:
        raise NotImplementedError

    def get_snapshot_cached(self, cooperativa_id: UUID, ttl: Optional[float] = None) -> Snapshot[T]:
        val_ttl = ttl if ttl is not None else self._DEFAULT_SNAPSHOT_TTL
        agora = time.monotonic()

        with self._lock:
            hit = self._snapshot_cache.get(cooperativa_id)
            if hit and (agora - hit[0]) < val_ttl:
                return hit[1]

            # Remove entradas expiradas para evitar acúmulo de objetos Pydantic na RAM
            expirados = [k for k, (ts, _) in self._snapshot_cache.items() if (agora - ts) >= val_ttl]
            for k in expirados:
                self._snapshot_cache.pop(k, None)

            while len(self._snapshot_cache) >= self._MAX_CACHE_ENTRIES:
                oldest_key = next(iter(self._snapshot_cache))
                self._snapshot_cache.pop(oldest_key, None)

        snap = self.get_snapshot(cooperativa_id)

        with self._lock:
            self._snapshot_cache[cooperativa_id] = (agora, snap)
        return snap
