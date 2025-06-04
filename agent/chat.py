import json
import logging
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from db.database import SessionLocal
from config.main import search_memories, search_similar_memories, get_embedding
from tools.memory_tools import process_memory_response
from config.cli import (
    display_welcome, display_user_input, display_memories,
    display_assistant_response, display_error, display_streaming_content
)
from config.config import CHAT_MODEL, MEMORY, THINK_MODE, get_config_section, VALID_CATEGORIES
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, with improved error handling and logging."""
    # Import valid categories from config
    from config.config import VALID_CATEGORIES
    
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
                    # Extract only the response content without the memory part
                    content = data.get('content', '')
                    memory = data.get('memory')
                    
                    # Process memory if it exists
                    if memory and isinstance(memory, str):
                        for category in VALID_CATEGORIES:
                            prefix = f"{category}: "
                            if memory.lower().startswith(prefix):
                                fact = memory[len(prefix):].strip()
                                data['memory'] = f"{category}: {fact}"
                                data['keyword'] = category
                                break
                    
                    # Process content
                    if isinstance(content, str):
                        # Remove any memory prefix from content if it exists
                        content_parts = content.split('\nResponse:', 1)
                        if len(content_parts) > 1:
                            content = content_parts[1].strip()
                        data['content'] = content
                    
                    return data
                logger.debug("Found JSON but missing required fields: %s", data)
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse JSON substring: %s", e)
                continue
        
        logger.warning("No valid JSON found in text, falling back to default structure")
        # If no valid JSON found, try to extract memory and response from text format
        
        # Try to find memory information by looking for valid category prefixes
        memory_match = None
        response_content = text
        
        # Look for valid category prefixes in the text
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            for category in VALID_CATEGORIES:
                prefix = f"{category}: "
                if line.lower().startswith(prefix):
                    memory_match = line
                    category_found = category
                    fact = line[len(prefix):].strip()
                    break
            if memory_match:
                break
        
        # If we found a valid memory, extract response
        if memory_match:
            # Look for response after memory
            response_parts = text.split('\nResponse:', 1)
            if len(response_parts) > 1:
                response_content = response_parts[1].strip()
            
            return {
                "content": response_content,
                "keyword": category_found,
                "memory": f"{category_found}: {fact}"
            }
        
        # If no memory found, return default
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
     * special (特别的经历、发现或感受，如"今天特别开心"、"发现了一家很棒的店"等)
     * insight (用户的思考、洞见、观点，如对人生的思考、对某个领域的深度见解等)
   - 'memory': MUST be in the format "<category>: <fact>" where category is one of the above categories in lowercase, or null if no fact to store
     Examples:
     - "preference: User likes Shiba Inu dogs"
     - "purchase: User bought a notebook last week"
     - "location: User lives in Beijing"
     - "schedule: User has a meeting on Monday at 2pm"
     - "special: User felt extremely happy today because of the sunny weather"
     - "special: User discovered a great hidden restaurant in the old town"
     - "insight: User believes that true innovation comes from combining different fields of knowledge"
     - "insight: User thinks continuous learning is key to personal growth"

     Only extract real facts about the user, ignore questions or casual chat.
When unsure, set memory to null rather than storing uncertain information."""

async def format_user_prompt(user_input: str, db: Session) -> str:
    # Get embedding for the current user input
    query_embedding = await get_embedding(user_input)
    
    # Search for relevant memories using vector similarity
    relevant_memories = await search_similar_memories(
        query=user_input,
        db=db,
        limit=5,  # Adjust this number based on needs
        ef_search=100  # Adjust based on performance needs
    )
    
    # Format the memories into context
    memory_context = ""
    if relevant_memories:
        memory_context = "\n\nRelevant memories based on semantic similarity:\n"
        for mem in relevant_memories:
            memory_context += f"- {mem.content} (Keyword: {mem.keyword})\n"
    else:
        memory_context = "\n\nNo relevant memories found for this query."

    return f"""User message: '{user_input}'
{memory_context}

Based on our conversation, extract any new factual information about the user and respond appropriately.
Remember to categorize any memory with the correct prefix (preference:, purchase:, etc).
If no new facts are shared, set memory to null."""

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

            # Initialize messages for Ollama
            ollama_messages = []
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    ollama_messages.append({
                        "role": "user",
                        "content": await format_user_prompt(msg["content"], self.db)
                    })
                else:
                    ollama_messages.append(msg)

            # Get completion from Ollama
            response = await self.client.chat.completions.create(
                model=CHAT_MODEL["name"],
                messages=ollama_messages,
                stream=True,
                **{k: v for k, v in CHAT_MODEL["parameters"].items() if k != 'stream'}
            )

            # Create a wrapper for the stream that captures think content
            async def stream_wrapper():
                nonlocal self
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk

            # Display streaming content and get full response
            full_response, think_content = await display_streaming_content(
                content_stream=stream_wrapper(),
                prefix=THINK_MODE['prefix']
            )

            # Process the response
            response_data = extract_json_from_text(full_response)

            if response_data:
                # Store memory if present
                memory = response_data.get("memory")
                fact = None
                category = None
                
                if memory and ': ' in memory:
                    # Split memory into category and fact
                    category, fact = memory.split(': ', 1)
                    
                    # Store in database
                    await process_memory_response(
                        memory=memory,
                        db=self.db,
                        user_id=self.current_user_id
                    )

                content = response_data.get("content", "")

                # Add assistant's response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content
                })

                # Display the response with think content
                display_assistant_response(
                    content,
                    fact,
                    category,
                    think_content
                )
            else:
                display_error("Failed to parse assistant response")

        except Exception as e:
            logger.error(f"Error during chat: {str(e)}", exc_info=True)
            display_error(str(e))

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