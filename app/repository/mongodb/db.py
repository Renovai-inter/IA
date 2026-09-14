from pymongo import MongoClient

def get_mongo_conn(dsn: str) -> MongoClient:
    return MongoClient(dsn)['mongo_renovaidb']