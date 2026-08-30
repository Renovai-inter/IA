from abc import ABC

from psycopg_pool import ConnectionPool

class Repository[T](ABC):
    def __init__(self, db: ConnectionPool):
        self._db = db