from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from app.repository.base import ConnectionFactory


class PostgresConnectionFactory(ConnectionFactory[ConnectionPool]):
    def __init__(self, dsn: str):
        super().__init__(dsn)

    def connect(self) -> ConnectionPool:
        return ConnectionPool(
            conninfo=self.dsn,
            min_size=1,
            max_size=10,
            max_idle=300,
            max_lifetime=1800,
            check=ConnectionPool.check_connection,
            kwargs={"row_factory": dict_row}
        )

    def close(self, conn) -> None:
        conn.close()