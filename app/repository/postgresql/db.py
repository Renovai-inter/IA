from psycopg_pool import ConnectionPool

def build_postgres_pool(dsn: str) -> ConnectionPool:
    return ConnectionPool(dsn)