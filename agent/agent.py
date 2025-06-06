from dataclasses import dataclass, field
from typing import List, Any, Protocol, Optional, Dict, AsyncIterator
from textwrap import dedent
import logging
import json
import os
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from db.database import SessionLocal
from config.main import search_memories, search_similar_memories, get_embedding
from tools.memory_tools import MemoryTools
from config.cli import (
    display_welcome, display_user_input, display_memories,
    display_assistant_response, display_error, display_streaming_content
)
from config.config import CHAT_MODEL, MEMORY, THINK_MODE, get_config_section, VALID_CATEGORIES
from rich.console import Console
from jarvis_mcp.mcp_client import MCPToolsManager
import re

logger = logging.getLogger(__name__)
console = Console()

class ChatModel(Protocol):
    """Protocol for chat models"""
    async def generate_response(self, messages: List[Dict[str, str]], stream: bool = True) -> AsyncIterator[str]: ...
    async def initialize(self) -> bool: ...

class OpenAIChat(ChatModel):
    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI chat model with configuration.
        Args:
            model: Optional model override
            **kwargs: Additional parameters to override config
        """
        # Get model configuration
        self.model_id = model or CHAT_MODEL["name"]
        
        # Build config from CHAT_MODEL parameters and any overrides
        self.config = {
            **CHAT_MODEL["parameters"],  # Base configuration from config.py
            **kwargs  # Override with any provided parameters
        }
        
        # API configuration
        self.api_base = CHAT_MODEL["base_url"]
        self.api_key = CHAT_MODEL["api_key"]
        self.client = None

    async def initialize(self) -> bool:
        """Initialize the OpenAI client with configuration."""
        # Initialize client with API configuration
        client_kwargs = {}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
            
        self.client = AsyncOpenAI(**client_kwargs)
        
        try:
            await self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "system", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {str(e)}")
            return False

    async def generate_response(self, messages: List[Dict[str, str]], stream: bool = True) -> AsyncIterator[str]:
        """Generate response from the model."""
        # Remove stream from config if it exists to avoid duplicate parameters
        config = self.config.copy()
        config.pop('stream', None)
        
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=stream,
            **config
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk

class AnthropicChat(ChatModel):
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.config = kwargs
        # TODO: Implement Anthropic chat integration

    async def initialize(self) -> bool:
        # TODO: Implement initialization
        return True

    async def generate_response(self, messages: List[Dict[str, str]], stream: bool = True) -> AsyncIterator[str]:
        # TODO: Implement response generation
        yield "Not implemented"

class MessageFormatter:
    def __init__(self, db: Session, custom_instructions: Optional[str] = None):
        self.db = db
        self.custom_instructions = custom_instructions

    async def format_user_prompt(self, user_input: str, user_id: str) -> str:
        query_embedding = await get_embedding(user_input)
        # First try to find exact keyword matches
        keyword_memories = await search_memories(
            keyword="preference",  # Since we're looking for preferences
            db=self.db
        )
        
        # Then find semantically similar memories
        similar_memories = await search_similar_memories(
            query=user_input,
            db=self.db,
            limit=5,
            ef_search=100,
            user_id=user_id
        )
        
        # Combine and deduplicate memories
        all_memories = []
        seen_contents = set()
        
        # Add keyword matches first
        for mem in keyword_memories:
            if mem.content not in seen_contents:
                all_memories.append(mem)
                seen_contents.add(mem.content)
        
        # Add similar matches
        for mem in similar_memories:
            if mem.content not in seen_contents:
                all_memories.append(mem)
                seen_contents.add(mem.content)
        
        # Format the memories into context
        memory_context = ""
        if all_memories:
            memory_context = "\n\nRelevant memories:\n"
            for mem in all_memories:
                memory_context += f"- {mem.content} (Keyword: {mem.keyword})\n"
        else:
            memory_context = "\n\nNo relevant memories found for this query."

        return f"""User message: '{user_input}'
{memory_context}

Based on our conversation and the memories above, respond naturally to the user's message.
If the user asks about something we have in memories, use that information in your response.
Extract any new factual information about the user and respond appropriately.
Remember to categorize any new memory with the correct prefix (preference:, purchase:, etc).
If no new facts are shared, set memory to null."""

    def get_system_prompt(self, user_id: str) -> str:
        base_prompt = f"""You are a helpful assistant. The user's ID is {user_id}.

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
When unsure, set memory to null rather than storing uncertain information.

IMPORTANT: Always format your response as a JSON object with the exact structure shown above.
Example response format:
{{
    "content": "Your natural response here",
    "keyword": "preference",
    "memory": "preference: User likes cats"
}}

If no new fact is learned, use:
{{
    "content": "Your natural response here",
    "keyword": "general",
    "memory": null
}}"""

        # Add custom instructions if provided
        if self.custom_instructions:
            base_prompt += f"\n\nAdditional Instructions:\n{self.custom_instructions}"

        return base_prompt

@dataclass
class Agent:
    name: str
    model: ChatModel
    instructions: str
    tools: List[Any] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    db: Optional[Session] = None
    current_user_id: Optional[str] = None
    message_formatter: Optional[MessageFormatter] = None
    custom_instructions: Optional[str] = None
    mcp: bool = False  # Controls whether MCP functionality is enabled
    mcp_manager: Optional[MCPToolsManager] = field(default=None, init=False)  # Initialized based on mcp flag

    def __post_init__(self):
        """Post initialization hook to dedent instructions and setup components"""
        self.instructions = dedent(self.instructions)
        if self.custom_instructions:
            self.custom_instructions = dedent(self.custom_instructions)
        self.db = SessionLocal()
        self.message_formatter = MessageFormatter(self.db, custom_instructions=self.custom_instructions)
        
        # Initialize MCP manager if enabled
        if self.mcp:
            self.mcp_manager = MCPToolsManager()
        
        # Initialize tools with database session
        for tool in self.tools:
            if hasattr(tool, 'set_db'):
                tool.set_db(self.db)
            if hasattr(tool, 'set_user_id'):
                tool.set_user_id(self.current_user_id)

    async def initialize(self) -> bool:
        """Initialize the agent and its components"""
        # Initialize chat model
        if not await self.model.initialize():
            return False
            
        # Initialize MCP manager and get tools if enabled
        if self.mcp and self.mcp_manager:
            logger.info("Initializing MCP tools...")
            if await self.mcp_manager.initialize():
                # Add MCP tools to the tools list
                mcp_tools = self.mcp_manager.get_tools()
                self.tools.extend(mcp_tools)
                logger.info(f"Added {len(mcp_tools)} MCP tools")
            else:
                logger.warning("Failed to initialize MCP manager")
                
        return True

    def setup_user(self):
        """Set up the user ID for the session"""
        default_user = get_config_section("user").get("default_id", "default_user")
        self.current_user_id = input(f"\nEnter your User ID (press Enter to use '{default_user}'): ").strip()
        if not self.current_user_id:
            self.current_user_id = default_user
        
        # Update user_id for all tools
        for tool in self.tools:
            if hasattr(tool, 'set_user_id'):
                tool.set_user_id(self.current_user_id)

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

    def extract_json_from_text(self, text: str) -> dict:
        """Extract JSON from text, with improved error handling and logging."""
        try:
            # First try to parse the whole text as JSON
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    # Validate required fields
                    content = data.get('content', '').strip()
                    memory = data.get('memory')
                    keyword = data.get('keyword', 'general')
                    
                    # Ensure memory format is correct if present
                    if memory and isinstance(memory, str):
                        if ': ' in memory:
                            category, fact = memory.split(': ', 1)
                            if category.lower() in VALID_CATEGORIES:
                                return {
                                    'content': content,
                                    'keyword': category.lower(),
                                    'memory': f"{category.lower()}: {fact.strip()}"
                                }
                    
                    # Return validated data
                    return {
                        'content': content,
                        'keyword': keyword,
                        'memory': memory if memory else None
                    }
                
            except json.JSONDecodeError:
                logger.debug("Full text is not valid JSON, trying to extract JSON portion")
                # Try to find JSON-like structure
                matches = re.finditer(r'{[^{}]*}', text)
                for match in matches:
                    try:
                        data = json.loads(match.group())
                        if isinstance(data, dict):
                            return self.extract_json_from_text(match.group())  # Recursively validate found JSON
                    except json.JSONDecodeError:
                        continue
            
            # If no valid JSON found, try to extract memory from text
            for category in VALID_CATEGORIES:
                prefix = f"{category}: "
                lines = text.split('\n')
                for line in lines:
                    if line.lower().startswith(prefix):
                        fact = line[len(prefix):].strip()
                        return {
                            'content': text.strip(),
                            'keyword': category,
                            'memory': f"{category}: {fact}"
                        }
            
            # Default response if no memory found
            return {
                'content': text.strip(),
                'keyword': 'general',
                'memory': None
            }
            
        except Exception as e:
            logger.error(f"Error in extract_json_from_text: {str(e)}", exc_info=True)
            return {
                'content': "I apologize, but I encountered an error processing the response.",
                'keyword': 'error',
                'memory': None
            }

    async def process_memory(self, response_data: dict):
        """Process and store memory if present in the response"""
        memory = response_data.get("memory")
        if memory and ': ' in memory:
            category, fact = memory.split(': ', 1)
            # Find memory tool
            memory_tool = next((tool for tool in self.tools if isinstance(tool, MemoryTools)), None)
            if memory_tool:
                fact, category = await memory_tool.process_memory_response(
                    memory=memory,
                    user_id=self.current_user_id,
                    db=self.db
                )
                return fact, category
        return None, None

    async def process_chat(self, user_input: str):
        """Process a single chat interaction"""
        try:
            # Add user message to history
            formatted_prompt = await self.message_formatter.format_user_prompt(user_input, self.current_user_id)
            self.conversation_history.append({
                "role": "user",
                "content": formatted_prompt
            })

            # Get system prompt with custom instructions
            system_prompt = self.message_formatter.get_system_prompt(self.current_user_id)
            
            # Add tool instructions if MCP is enabled
            if self.mcp and self.mcp_manager:
                tools = self.mcp_manager.get_tools()
                tool_instructions = "\n\nAvailable tools:\n"
                for tool in tools:
                    tool_instructions += f"- {tool.name}: {tool.description}\n"
                system_prompt += tool_instructions
            
            # Combine instructions with system prompt
            combined_instructions = f"{self.instructions}\n\n{system_prompt}"

            # Get completion from the model
            response_stream = self.model.generate_response(
                messages=[
                    {"role": "system", "content": combined_instructions},
                    *self.conversation_history
                ]
            )

            # Display streaming content and get full response
            full_response, think_content = await display_streaming_content(
                content_stream=response_stream,
                prefix=THINK_MODE['prefix']
            )

            # Process the response
            response_data = self.extract_json_from_text(full_response)

            if response_data:
                content = response_data.get("content", "")

                # Process tool calls if MCP is enabled
                if self.mcp and self.mcp_manager and "tool_calls" in response_data:
                    tool_calls = response_data.get("tool_calls", [])
                    for tool_call in tool_calls:
                        try:
                            tool_result = await self.mcp_manager.call_tool(
                                tool_name=tool_call["name"],
                                tool_arguments=tool_call["arguments"],
                                tool_id=tool_call.get("id", "default")
                            )
                            
                            # Add tool result to conversation history
                            self.conversation_history.append(tool_result)
                            
                            # Update content with tool result
                            content = f"{content}\n\n工具调用结果：\n{tool_result['content']}"
                            
                        except Exception as e:
                            logger.error(f"Error calling tool {tool_call['name']}: {str(e)}")
                            content = f"{content}\n\n工具调用错误：{str(e)}"

                # Process memory if present
                fact, category = await self.process_memory(response_data)

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

    async def cleanup(self):
        """Clean up resources used by the agent"""
        if self.db:
            self.db.close()
            
        # Clean up MCP resources if enabled
        if self.mcp and self.mcp_manager:
            logger.info("Cleaning up MCP resources...")
            await self.mcp_manager.cleanup()

    async def start(self):
        """Start the agent's chat loop"""
        display_welcome()
        
        try:
            # Initialize the agent
            if not await self.initialize():
                return

            # Set up user
            self.setup_user()
            
            # Main chat loop
            while True:
                # Get user input
                user_input = display_user_input(self.current_user_id)
                
                # Handle commands
                if await self.handle_command(user_input):
                    break
                
                # Process chat
                await self.process_chat(user_input)
                
        except KeyboardInterrupt:
            print("\nChat ended by user.")
        finally:
            await self.cleanup() 