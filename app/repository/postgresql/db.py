from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

def build_postgres_pool(dsn: str) -> ConnectionPool:
    return ConnectionPool(dsn, kwargs={"row_factory": dict_row})