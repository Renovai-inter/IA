from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

def build_postgres_pool(dsn: str) -> ConnectionPool:
    return ConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=10,
        max_idle=300,
        max_lifetime=1800,
        check=ConnectionPool.check_connection,
        kwargs={"row_factory": dict_row}
    )