import psycopg2
from app.core.config import Settings

settings = Settings()

def get_conn():
    """Estabelece conexão com o banco PostgreSQL"""
    return psycopg2.connect(settings.DATABASE_URL)