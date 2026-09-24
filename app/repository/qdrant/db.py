from qdrant_client import QdrantClient

from app.repository.base import ConnectionFactory


class QdrantConnectionFactory(ConnectionFactory[QdrantClient]):
    def __init__(self, dsn: str, api_key = str):
        super().__init__(dsn)
        self._api_key = api_key
        self._client: QdrantClient | None = None

    def connect(self) -> QdrantClient:
        self._client = QdrantClient(url=self.dsn, api_key=self._api_key)

    def close(self, conn: QdrantClient) -> None:
        self._client.close()