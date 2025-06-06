import asyncio
import logging
import os
from config.config import LOGGING
from agent.agent import Agent, OpenAIChat
from tools.memory_tools import MemoryTools
from tools.crawl_tools import CrawlTools

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING["level"]),
    format=LOGGING["format"],
    filename=LOGGING["file"]
)
logger = logging.getLogger(__name__)

agent = Agent(
    name="Jarvis",
    model=OpenAIChat(
        model=os.getenv("OPENAI_MODEL_ID"),
        temperature=0.7,
        max_tokens=2000
    ),
    instructions="""
    I am Jarvis, your AI assistant. I can help you with various tasks using my available tools.
    I will follow your instructions and provide helpful responses.
    """,
    tools=[
        MemoryTools(),
        CrawlTools()
    ]
)

if __name__ == "__main__":
    asyncio.run(agent.start()) 