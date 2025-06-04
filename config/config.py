import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# User Configuration
USER = {
    "default_id": os.getenv("DEFAULT_USER_ID", "Vandee")
}

# Model Configuration
CHAT_MODEL = {
    "name": os.getenv("CHAT_MODEL", "qwen3:14b"),
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "api_key": os.getenv("OLLAMA_API_KEY", None),
    "parameters": {
        "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("CHAT_TOP_P", "0.9")),
        "max_tokens": int(os.getenv("CHAT_MAX_TOKENS", "2000")),
        "stream": True
    }
}

# Embedding Model Configuration
EMBEDDING = {
    "model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    "base_url": os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1"),
    "api_key": os.getenv("EMBEDDING_API_KEY", None),
    "dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
}

# Database Configuration
DATABASE = {
    "url": os.getenv("DATABASE_URL", "sqlite:///./jarvis.db"),
    "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
    "echo": os.getenv("DB_ECHO", "false").lower() == "true"
}

# Memory Settings
MEMORY = {
    "user_memory_limit": int(os.getenv("USER_MEMORY_LIMIT", "5")),
    "similar_memory_limit": int(os.getenv("SIMILAR_MEMORY_LIMIT", "3")),
    "ef_search": int(os.getenv("EF_SEARCH", "100")),
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
}

# Valid memory categories
VALID_CATEGORIES = {
    'preference',  # 用户偏好，如喜欢的食物、动物等
    'purchase',    # 购物相关
    'location',    # 位置相关，如居住地、工作地等
    'schedule',    # 日程相关，如约会、会议等
    'contact',     # 联系人相关
    'personal',    # 个人信息，如职业、爱好等
    'special',     # 特别的经历、发现或感受
    'insight'      # 用户的思考、洞见、观点
}

# Logging Configuration
LOGGING = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": os.getenv(
        "LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ),
    "file": os.getenv("LOG_FILE", "jarvis.log")
}

# API Configuration (if needed in the future)
API = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("API_DEBUG", "false").lower() == "true"
}

# Think mode Configuration
THINK_MODE = {
    "enabled": True,
    "prefix": "🤔",
    "buffer_size": int(os.getenv("THINK_BUFFER_SIZE", "50")),
    "update_interval": float(os.getenv("THINK_UPDATE_INTERVAL", "0.1"))
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        "user": USER,
        "chat_model": CHAT_MODEL,
        "embedding": EMBEDDING,
        "database": DATABASE,
        "memory": MEMORY,
        "logging": LOGGING,
        "api": API
    }

# Convenience function to get a specific config section
def get_config_section(section: str) -> Dict[str, Any]:
    """Get a specific configuration section."""
    config = get_config()
    return config.get(section, {}) 