import asyncio
import logging
from config.config import LOGGING
from agent.chat import start_chat

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING["level"]),
    format=LOGGING["format"],
    filename=LOGGING["file"]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    asyncio.run(start_chat()) 