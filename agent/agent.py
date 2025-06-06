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
    display_assistant_response, display_error, display_streaming_content,
    display_tool_calling, display_thinking
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
        return f"User message: '{user_input}'\n\nRespond naturally to the user's message."

    def get_system_prompt(self, user_id: str) -> str:
        base_prompt = f"""You are a helpful AI assistant that can both engage in natural conversations and access memory capabilities. The user's ID is {user_id}.

Core Functions:
1. Natural Conversation
- Engage in normal dialogue naturally and professionally
- Provide direct, clear answers
- Be helpful and proactive in suggesting solutions

2. Memory Operations (via function calls)
- search_memories_by_keyword: Find memories by category
- search_memories_by_similarity: Find contextually similar memories
- store_memory: Save new important information
- get_memory_context: Get relevant memory context

Memory Categories:
* preference - User preferences (food, colors, etc.)
* purchase - Shopping and purchase history
* location - Location-related information
* schedule - Calendar and scheduling information
* contact - Contact and relationship information
* personal - Personal information (job, hobbies, etc.)
* special - Special experiences or discoveries
* insight - User's thoughts, insights, and viewpoints

Response Guidelines:
1. When using memories:
   - Include ALL relevant retrieved memories in your response
   - Present related information together (e.g., all preferences, all schedules)
   - If memories conflict, acknowledge all and note the contradiction
2. For general conversation:
   - Be direct and natural in your responses
   - Focus on solving the user's needs
   - Don't be overly verbose

Response Format:
{{
    "content": "Your natural response (including ALL relevant memories if any)",
    "keyword": "most_relevant_category_or_general",
    "memory": null  // Memory operations handled via function calls
}}

IMPORTANT: Always use function calls for memory operations. Respond naturally to general queries."""

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
    mcp: bool = False
    mcp_manager: Optional[MCPToolsManager] = field(default=None, init=False)
    max_chat_history: int = field(default=5)  # Default history limit
    system_prompt: Optional[str] = field(default=None, init=False)  # Store system prompt as metadata

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
        
        # Initialize memory tools
        memory_tool = MemoryTools()
        memory_tool.set_db(self.db)
        memory_tool.set_user_id(self.current_user_id)
        self.tools.append(memory_tool)
        
        # Initialize other tools with database session
        for tool in self.tools:
            if hasattr(tool, 'set_db'):
                tool.set_db(self.db)
            if hasattr(tool, 'set_user_id'):
                tool.set_user_id(self.current_user_id)

    def _trim_conversation_history(self):
        """Trim conversation history to the maximum allowed length"""
        if len(self.conversation_history) > self.max_chat_history * 2:  # *2 because each exchange has user + assistant message
            self.conversation_history = self.conversation_history[-(self.max_chat_history * 2):]

    def _get_messages_with_system_prompt(self) -> List[Dict[str, str]]:
        """Get full message list including system prompt"""
        if not self.system_prompt:
            # Generate system prompt if not already generated
            base_prompt = self.message_formatter.get_system_prompt(self.current_user_id)
            self.system_prompt = f"{self.instructions}\n\n{base_prompt}"
            
        return [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]

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
            
            # Trim conversation history if needed
            self._trim_conversation_history()

            # Get messages including system prompt
            llm_messages = self._get_messages_with_system_prompt()

            # Prepare tools configuration
            tools_config = []
            
            # Add memory tools
            memory_tool = next((tool for tool in self.tools if isinstance(tool, MemoryTools)), None)
            if memory_tool:
                tools_config.extend(memory_tool.function_definitions)
            
            # Add MCP tools if available
            if self.mcp and self.mcp_manager:
                mcp_tools = self.mcp_manager.format_tools_for_llm()
                if mcp_tools:
                    tools_config.extend(mcp_tools)

            # LLM Interaction Loop
            MAX_TOOL_CALL_ITERATIONS = 5
            current_iteration = 0

            while current_iteration < MAX_TOOL_CALL_ITERATIONS:
                current_iteration += 1
                logger.info(f"Starting iteration {current_iteration}")

                # First, make a non-streaming call to check for tool usage
                llm_kwargs = {
                    "model": CHAT_MODEL["name"],
                    "messages": llm_messages,
                    "tools": tools_config,
                    "tool_choice": "auto",
                    "stream": False  # Explicitly set non-streaming for tool calls
                }
                
                # Add model parameters from config (except stream)
                for k, v in CHAT_MODEL["parameters"].items():
                    if k != 'stream':
                        llm_kwargs[k] = v

                try:
                    with display_thinking():
                        response = await self.model.client.chat.completions.create(**llm_kwargs)
                        assistant_message = response.choices[0].message

                    # If no tool calls, we're done - get the final streaming response
                    if not hasattr(assistant_message, 'tool_calls') or not assistant_message.tool_calls:
                        # Make the final streaming call
                        streaming_kwargs = llm_kwargs.copy()
                        streaming_kwargs["stream"] = True
                        streaming_response = await self.model.client.chat.completions.create(**streaming_kwargs)
                        
                        full_llm_response_text, current_think_content = await display_streaming_content(
                            streaming_response,
                            prefix=THINK_MODE["prefix"],
                            is_thinking=THINK_MODE["enabled"]
                        )
                        
                        # Add final response to history
                        final_message = {
                            "role": "assistant",
                            "content": full_llm_response_text
                        }
                        self.conversation_history.append(final_message)
                        
                        # Process the final response
                        response_data = self.extract_json_from_text(full_llm_response_text)
                        if response_data:
                            content = response_data.get("content", "")
                            display_assistant_response(
                                content,
                                None,
                                None,
                                current_think_content
                            )
                        break

                    # Process tool calls
                    tool_calls = assistant_message.tool_calls
                    logger.info(f"Processing {len(tool_calls)} tool calls")
                    display_tool_calling(tool_calls)

                    # Add assistant's tool call message to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [t.model_dump(exclude_none=True) for t in tool_calls]
                    })

                    # Process each tool call and add results to conversation
                    tool_results = []
                    for tool_call in tool_calls:
                        try:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            tool_id = tool_call.id

                            # Handle memory tool calls
                            if memory_tool and tool_name in [t["function"]["name"] for t in memory_tool.function_definitions]:
                                tool_method = getattr(memory_tool, tool_name)
                                tool_result = await tool_method(**tool_args)
                                
                                tool_response = {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": json.dumps(tool_result)
                                }
                                tool_results.append(tool_response)
                                
                            else:
                                # Handle MCP tool calls
                                tool_result = await self.mcp_manager.call_tool(
                                    tool_name=tool_name,
                                    tool_arguments=tool_args,
                                    tool_id=tool_id
                                )
                                tool_results.append(tool_result)
                            
                        except Exception as e:
                            logger.error(f"Tool call failed for {tool_name}: {e}")
                            error_response = {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": json.dumps({
                                    "error": "execution_error",
                                    "message": str(e)
                                })
                            }
                            tool_results.append(error_response)

                    # Add all tool results to conversation history at once
                    self.conversation_history.extend(tool_results)
                    
                    # Update messages for next iteration
                    llm_messages = self.conversation_history.copy()

                except Exception as e:
                    logger.error(f"Error during LLM interaction: {str(e)}", exc_info=True)
                    display_error(f"LLM API call failed: {str(e)}")
                    return

            # Handle max iterations reached
            if current_iteration >= MAX_TOOL_CALL_ITERATIONS:
                logger.warning("Max tool call iterations reached")
                display_error("I apologize, but I seem to be having trouble processing your request. Could you please try rephrasing it?")

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