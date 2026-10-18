from neo4j import Driver, GraphDatabase

from app.repository.base import ConnectionFactory


class Neo4jConnectionFactory(ConnectionFactory[Driver]):
    def __init__(self, dsn: str, user: str = 'neo4j', password: str = ''):
        super().__init__(dsn)
        self._auth = (user, password)
        self._driver: Driver | None = None

    def connect(self) -> Driver:
        self._driver = GraphDatabase.driver(self.dsn, auth=self._auth)
        return self._driver

    def close(self, conn: Driver | None = None) -> None:
        target = conn or self._driver
        if target:
            target.close()
