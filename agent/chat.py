import json
import re
import logging
import asyncio
from openai import AsyncOpenAI
from db.database import SessionLocal
from config.main import search_memories, search_similar_memories, get_user_specific_memories
from tools.memory_tools import format_memory_context, process_memory_response
from config.cli import (
    display_welcome, display_user_input, display_memories,
    display_assistant_response, display_error, display_streaming_content
)
from config.config import LOGGING, CHAT_MODEL, MEMORY, THINK_MODE, get_config_section
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, with improved error handling and logging."""
    try:
        # First try to parse the whole text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Full text is not valid JSON, trying to extract JSON portion")
        
        # Try to find a JSON-like structure
        json_text = text.strip()
        logger.debug("Attempting to extract JSON from: %s", json_text[:200])  # Log first 200 chars
        
        # Find all pairs of braces
        brace_pairs = []
        stack = []
        for i, char in enumerate(json_text):
            if char == '{':
                stack.append(i)
            elif char == '}' and stack:
                start = stack.pop()
                brace_pairs.append((start, i))
        
        # Try each brace pair from longest to shortest
        for start, end in sorted(brace_pairs, key=lambda x: x[1] - x[0], reverse=True):
            try:
                json_str = json_text[start:end + 1]
                data = json.loads(json_str)
                if isinstance(data, dict):
                    if 'content' in data and 'keyword' in data:
                        return data
                    logger.debug("Found JSON but missing required fields: %s", data)
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse JSON substring: %s", e)
                continue
        
        logger.warning("No valid JSON found in text, falling back to default structure")
        # If no valid JSON found, create one from the text
        return {
            "content": text.strip(),
            "keyword": "general",
            "memory": None
        }
        
    except Exception as e:
        logger.error("Error in extract_json_from_text: %s", e, exc_info=True)
        # In case of any error, return a safe default
        return {
            "content": "I apologize, but I encountered an error processing the response. Could you please rephrase your question?",
            "keyword": "error",
            "memory": None
        }

def get_system_prompt(user_id: str) -> str:
    return f"""You are a helpful assistant. The user's ID is {user_id}.

Your tasks:
1. Respond naturally to the user's message
2. Extract and structure important information about the user
3. Return a JSON with three fields:
   - 'content': your response to the user
   - 'keyword': a category tag from the following list ONLY:
     * preference (用户偏好，如喜欢的食物、动物等)
     * purchase (购物相关)
     * location (位置相关，如居住地、工作地等)
     * schedule (日程相关，如约会、会议等)
     * contact (联系人相关)
     * personal (个人信息，如职业、爱好等)
   - 'memory': a structured fact about the user in third person, or null if no fact to store
     Format: "<category>: <fact>"
     Examples:
     - "preference: User likes Shiba Inu dogs"
     - "purchase: User bought a notebook last week"
     - "location: User lives in Beijing"
     - "schedule: User has a meeting on Monday at 2pm"

Only extract real facts about the user, ignore questions or casual chat.
When unsure, set memory to null rather than storing uncertain information."""

def format_user_prompt(user_input: str, memory_context: str) -> str:
    return f"""User message: '{user_input}'
{memory_context}

Based on our conversation, extract any new factual information about the user and respond appropriately.
Remember to categorize any memory with the correct prefix (preference:, purchase:, etc).
If no new facts are shared, set memory to null.
"""

class ChatSession:
    def __init__(self):
        self.db = SessionLocal()
        self.client = None
        self.conversation_history = []
        self.current_user_id = None

    async def initialize(self):
        """Initialize the chat session with OpenAI client and test connection"""
        self.client = AsyncOpenAI(
            base_url=CHAT_MODEL["base_url"],
            api_key=CHAT_MODEL["api_key"]
        )
        
        try:
            await self.client.chat.completions.create(
                model=CHAT_MODEL["name"],
                messages=[{"role": "system", "content": "test"}],
                max_tokens=1
            )
        except Exception as e:
            logger.error(f"Failed to connect to model service: {str(e)}")
            display_error("Failed to connect to model service. Please ensure Ollama is running.")
            return False
        return True

    def setup_user(self):
        """Set up the user ID for the session"""
        default_user = get_config_section("user").get("default_id", "default_user")
        self.current_user_id = input(f"\nEnter your User ID (press Enter to use '{default_user}'): ").strip()
        if not self.current_user_id:
            self.current_user_id = default_user

    async def handle_command(self, user_input: str) -> bool:
        """Handle special commands and return True if a command was handled"""
        if user_input.lower() == '/exit':
            print("\nGoodbye!")
            return True
        elif user_input.lower().startswith('/search '):
            keyword = user_input[8:].strip()
            memories = await search_memories(keyword=keyword, db=self.db)
            display_memories(memories, "keyword")
            return True
        elif user_input.lower().startswith('/similar '):
            query = user_input[9:].strip()
            memories = await search_similar_memories(
                query=query, 
                db=self.db, 
                limit=MEMORY["similar_memory_limit"], 
                ef_search=MEMORY["ef_search"]
            )
            display_memories(memories, "similar")
            return True
        return False

    async def process_chat(self, user_input: str):
        """Process a single chat interaction"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })

            # 1. Load user-specific memories
            user_specific_memories_list = await get_user_specific_memories(
                user_id=self.current_user_id, 
                db=self.db, 
                limit=MEMORY["user_memory_limit"]
            )
            
            # 2. Find general similar memories (semantic search)
            similar_memories_list = await search_similar_memories(
                query=user_input, 
                db=self.db, 
                limit=MEMORY["similar_memory_limit"], 
                ef_search=MEMORY["ef_search"]
            )
            
            # 3. Format memory context
            combined_memory_context = await format_memory_context(
                user_specific_memories_list,
                similar_memories_list
            )

            # Construct messages for chat
            ollama_messages = [
                {
                    "role": "system",
                    "content": get_system_prompt(self.current_user_id)
                }
            ]

            # Add recent conversation history
            history_to_include = self.conversation_history[-4:] 
            ollama_messages.extend(history_to_include)

            # Add current user input and memory context
            ollama_messages.append({
                "role": "user",
                "content": format_user_prompt(user_input, combined_memory_context)
            })

            # Log debug information if enabled
            if LOGGING["level"] == "DEBUG":
                self._log_debug_info(user_specific_memories_list, similar_memories_list, ollama_messages)

            # Initialize current_think_content as instance variable
            self.current_think_content = ""

            # Get model response with streaming
            stream = await self.client.chat.completions.create(
                model=CHAT_MODEL["name"],
                messages=ollama_messages,
                stream=True,
                **{k: v for k, v in CHAT_MODEL["parameters"].items() if k != 'stream'}
            )

            # Create a wrapper for the stream that captures think content
            async def stream_wrapper():
                nonlocal self
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk

            # Display streaming content and get full response
            full_response, think_content = await display_streaming_content(
                content_stream=stream_wrapper(),
                prefix=THINK_MODE['prefix']
            )

            # Save the think content
            self.current_think_content = think_content

            # Process the complete response
            processed_response = await self._process_response(
                full_response,
                user_input, 
                user_specific_memories_list, 
                similar_memories_list, 
                ollama_messages
            )
            
            if processed_response:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": processed_response["content"]
                })

        except Exception as e:
            logger.error(f"Error during chat: {str(e)}", exc_info=True)
            display_error(str(e))

    def _log_debug_info(self, user_memories, similar_memories, messages):
        """Log debug information about the current chat state"""
        logger.debug("Memory context:")
        if user_memories:
            logger.debug("User memories:")
            for mem in user_memories:
                logger.debug(f"- {mem.content} (Keyword: {mem.keyword})")
        if similar_memories:
            logger.debug("Similar memories:")
            for mem in similar_memories:
                logger.debug(f"- {mem.content} (Keyword: {mem.keyword})")

        logger.debug("Sending messages to model:")
        for msg in messages:
            logger.debug(f"- Role: {msg['role']}, Content: {msg['content'][:100]}...")

    async def _process_response(self, response_text: str, user_input: str, user_memories: list, similar_memories: list, messages: list):
        """Process the model's response and handle any errors"""
        try:
            if not response_text:
                logger.error("Empty response text")
                raise ValueError("Empty response from model")
                
            logger.debug("Raw response from model: %s", response_text)
            
            data = extract_json_from_text(response_text)
            logger.debug("Extracted data: %s", str(data))
            
            if not isinstance(data, dict):
                raise ValueError(f"Invalid response format - expected dict, got {type(data)}")
                
            assistant_response = data.get('content', response_text)
            keyword = data.get('keyword', 'general')
            memory = data.get('memory')
            
            logger.debug("Processed response: content=%s, keyword=%s, memory=%s", 
                       assistant_response, keyword, memory)
            
            # Process and store memory if present
            fact, category = await process_memory_response(memory, self.current_user_id, self.db)
            
            # Prepare debug info if enabled
            debug_info = None
            if LOGGING["level"] == "DEBUG":
                debug_info = {
                    "Memory Context": {
                        "User Memories": [{"content": mem.content, "keyword": mem.keyword} 
                                        for mem in user_memories] if user_memories else [],
                        "Similar Memories": [{"content": mem.content, "keyword": mem.keyword} 
                                           for mem in similar_memories] if similar_memories else []
                    },
                    "Model Interaction": {
                        "Input": messages[-1]["content"],
                        "Raw Response": response_text,
                        "Processed Response": {
                            "Content": assistant_response,
                            "Keyword": keyword,
                            "Memory": memory
                        }
                    }
                }
                debug_info = json.dumps(debug_info, indent=2, ensure_ascii=False)

            # Display the response using cli.py's display_assistant_response
            display_assistant_response(
                response=assistant_response,
                fact=fact,
                category=category,
                debug_info=debug_info,
                current_think_content=self.current_think_content
            )

            return data

        except Exception as e:
            logger.error(f"Error processing model response: {str(e)}", exc_info=True)
            display_error(f"Failed to process model response: {str(e)}")
            return None

    def cleanup(self):
        """Clean up resources used by the chat session"""
        if self.db:
            self.db.close()

async def start_chat():
    """Main function to start and manage a chat session"""
    display_welcome()
    
    chat_session = ChatSession()
    try:
        # Initialize the session
        if not await chat_session.initialize():
            return

        # Set up user
        chat_session.setup_user()
        
        # Main chat loop
        while True:
            # Get user input
            user_input = display_user_input(chat_session.current_user_id)
            
            # Handle commands
            if await chat_session.handle_command(user_input):
                break
            
            # Process chat
            await chat_session.process_chat(user_input)
            
    except KeyboardInterrupt:
        print("\nChat ended by user.")
    finally:
        chat_session.cleanup() 