from pymongo import MongoClient
from core.config import settings, logger

def get_db_client():
	try:
		client = MongoClient(settings.mongo_url)
		db = client[settings.database_name]
		collection = db[settings.collection_name]
		logger.info("✅ MongoDB connected successfully (from db.py)")
		return client, db, collection
	except Exception as e:
		logger.error(f"❌ MongoDB connection failed: {e}")
		raise
