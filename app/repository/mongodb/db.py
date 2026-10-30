from __future__ import annotations

import logging
import certifi
from pymongo import MongoClient
from pymongo.database import Database

from app.repository.base import ConnectionFactory

log = logging.getLogger('renovai.mongodb')


class MongoConnectionFactory(ConnectionFactory[Database]):
    def __init__(self, dsn: str, db_name: str):
        super().__init__(dsn)
        self._db_name = db_name
        self._client: MongoClient | None = None

    def connect(self) -> Database:
        kwargs: dict = {}
        try:
            ca_file = certifi.where()
            if ca_file:
                kwargs['tlsCAFile'] = ca_file
        except Exception as exc:
            log.warning(f'Falha ao carregar certificados CA via certifi ({exc}).')

        kwargs.setdefault('serverSelectionTimeoutMS', 10000)
        kwargs.setdefault('connectTimeoutMS', 10000)

        self._client = MongoClient(self.dsn, **kwargs)
        return self._client[self._db_name]

    def close(self, conn: Database) -> None:
        if self._client:
            self._client.close()
