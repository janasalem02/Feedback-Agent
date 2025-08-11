import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:
	"""إعدادات النظام"""
    
	# MongoDB Configuration
	mongo_url: str = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
	database_name: str = os.getenv("DATABASE_NAME", "audio_db")
	collection_name: str = os.getenv("COLLECTION_NAME", "audio_files")
	criteria_collection: str = os.getenv("CRITERIA_COLLECTION", "client")
    
	# File Upload Configuration
	upload_folder: str = os.getenv("UPLOAD_FOLDER", "content")
	max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
    
	# Whisper Configuration
	whisper_model_name: str = os.getenv("WHISPER_MODEL", "large-v3")
	whisper_model_path: str = os.getenv("WHISPER_MODEL_PATH", "models/faster-whisper-large-v3")
	whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    
	# AI/LLM Configuration
	ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
	llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
	llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    
	# Analysis Configuration
	analysis_batch_size: int = int(os.getenv("ANALYSIS_BATCH_SIZE", "10"))
	max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

# Create settings instance
settings = Settings()

# Export logger for easy access
__all__ = ["settings", "logger"]
