from pymongo import MongoClient
from pymongo.database import Database

from app.repository.base import ConnectionFactory


class MongoConnectionFactory(ConnectionFactory[Database]):
    def __init__(self, dsn: str, db_name: str):
        super().__init__(dsn)
        self._db_name = db_name
        self._client: MongoClient | None = None

    def connect(self) -> Database:
        self._client = MongoClient(self.dsn)
        return self._client[self._db_name]

    def close(self, conn: Database) -> None:
        if self._client:
            self._client.close()