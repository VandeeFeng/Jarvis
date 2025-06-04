import json
import re
import logging
import asyncio
from typing import Optional
from urllib.parse import urlparse, urljoin
from contextlib import AsyncExitStack
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from db.database import SessionLocal
from config.main import search_memories, search_similar_memories, get_user_specific_memories
from tools.memory_tools import format_memory_context, process_memory_response
from config.cli import (
    display_welcome, display_user_input, display_memories,
    display_assistant_response, display_error, display_streaming_content
)
from config.config import LOGGING, CHAT_MODEL, MEMORY, THINK_MODE, get_config_section, MCP_SERVER
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
        self.mcp_config = MCP_SERVER
        self.mcp_sessions = {}  # Dictionary to store multiple MCP sessions
        self.mcp_exit_stack = AsyncExitStack()
        self.mcp_tools = []  # Combined tools from all sessions
        self._sse_tasks = {}  # Dictionary to store SSE tasks for each server

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
            logger.info("Successfully connected to Ollama service.")
        except Exception as e:
            logger.error(f"Failed to connect to model service: {str(e)}")
            display_error("Failed to connect to model service. Please ensure Ollama is running.")
            return False

        if self.mcp_config.get("enabled"):
            logger.info("MCP client is enabled. Attempting to connect to MCP servers...")
            server_type = self.mcp_config.get("type")
            
            if server_type == "sse":
                servers = self.mcp_config.get("servers", {})
                
                for server_name, server_config in servers.items():
                    try:
                        logger.info(f"Connecting to MCP SSE server: {server_name}")
                        server_url = server_config["url"]
                        
                        logger.debug(f"Creating SSE transport for {server_name} at {server_url}")
                        try:
                            # Add timeout for SSE connection
                            async with asyncio.timeout(10):  # 10 seconds timeout
                                sse_transport = await self.mcp_exit_stack.enter_async_context(
                                    sse_client(url=server_url)
                                )
                                read_stream, write_stream = sse_transport
                                session = await self.mcp_exit_stack.enter_async_context(
                                    ClientSession(read_stream, write_stream)
                                )
                                
                                # Initialize session with timeout
                                async with asyncio.timeout(5):  # 5 seconds timeout for initialization
                                    await session.initialize()
                                    
                                self.mcp_sessions[server_name] = session
                                
                                # Start SSE event listener for this server
                                self._sse_tasks[server_name] = asyncio.create_task(
                                    self._handle_sse_events(server_name, session)
                                )
                                
                                # Get tools from this server with timeout
                                async with asyncio.timeout(5):  # 5 seconds timeout for tool listing
                                    tools_response = await session.list_tools()
                                    if tools_response and tools_response.tools:
                                        # Add server name to each tool for identification
                                        for tool in tools_response.tools:
                                            tool.server_name = server_name
                                        self.mcp_tools.extend(tools_response.tools)
                                        logger.info(f"Added {len(tools_response.tools)} tools from {server_name}")
                                
                                logger.info(f"Successfully connected to {server_name}")
                            
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout while connecting to {server_name}")
                            continue
                        except Exception as e:
                            logger.error(f"Failed to establish SSE connection to {server_name}: {str(e)}", exc_info=True)
                            continue
                            
                    except Exception as e:
                        logger.error(f"Failed to connect to MCP server {server_name}: {str(e)}", exc_info=True)
                        display_error(f"Failed to connect to MCP server {server_name}")
                        continue
                
                if not self.mcp_sessions:
                    logger.error("Failed to connect to any MCP servers")
                    return False
                
                logger.info(f"Successfully connected to {len(self.mcp_sessions)} MCP servers")
                logger.info(f"Total available tools: {len(self.mcp_tools)}")
            else:
                logger.error(f"Unsupported MCP server type: {server_type}")
                display_error(f"Unsupported MCP server type: {server_type}")
                return False
        else:
            logger.info("MCP client is disabled in configuration.")
            
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

            # Prepare messages for the LLM
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

            if LOGGING["level"] == "DEBUG":
                self._log_debug_info(user_specific_memories_list, similar_memories_list, ollama_messages)

            self.current_think_content = ""
            full_response_content = ""

            # Prepare MCP tools for the LLM if MCP is enabled and sessions exist
            formatted_mcp_tools = []
            if self.mcp_sessions and self.mcp_tools:
                for tool in self.mcp_tools:
                    try:
                        parameters = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
                        if not isinstance(parameters, dict):
                            if isinstance(parameters, str):
                                parameters = json.loads(parameters)
                            else:
                                parameters = {"type": "object", "properties": {}}

                        formatted_mcp_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or f"Executes the {tool.name} tool.",
                                "parameters": parameters
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error formatting MCP tool {tool.name}: {e}", exc_info=True)

            # LLM Interaction Loop
            MAX_TOOL_CALL_ITERATIONS = 5
            current_iteration = 0

            while current_iteration < MAX_TOOL_CALL_ITERATIONS:
                current_iteration += 1

                llm_kwargs = {
                    "model": CHAT_MODEL["name"],
                    "messages": ollama_messages,
                    "stream": True,
                }
                for k, v in CHAT_MODEL["parameters"].items():
                    if k != 'stream':
                        llm_kwargs[k] = v

                if formatted_mcp_tools and current_iteration == 1:
                    llm_kwargs["tools"] = formatted_mcp_tools
                    llm_kwargs["tool_choice"] = "auto"
                    logger.debug("Sending request to LLM with MCP tools.")
                else:
                    logger.debug("Sending request to LLM without tools.")

                try:
                    # Get non-streamed response for tool check
                    llm_kwargs_for_tool_check = llm_kwargs.copy()
                    llm_kwargs_for_tool_check["stream"] = False

                    response_from_llm = await self.client.chat.completions.create(**llm_kwargs_for_tool_check)
                    assistant_message = response_from_llm.choices[0].message

                    if assistant_message.content or assistant_message.tool_calls:
                        ollama_messages.append(assistant_message.model_dump(exclude_none=True))

                    if assistant_message.tool_calls:
                        logger.info(f"LLM requested tool calls: {[tc.function.name for tc in assistant_message.tool_calls]}")
                        for tool_call in assistant_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_id = tool_call.id

                            # Find the tool and its server
                            tool = next((t for t in self.mcp_tools if t.name == tool_name), None)
                            if not tool:
                                logger.error(f"Tool {tool_name} not found in any MCP server")
                                ollama_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": f"Error: Tool {tool_name} not available"
                                })
                                continue

                            server_name = getattr(tool, 'server_name', None)
                            if not server_name or server_name not in self.mcp_sessions:
                                logger.error(f"Server not found for tool {tool_name}")
                                continue

                            session = self.mcp_sessions[server_name]
                            try:
                                tool_arguments = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing arguments for tool {tool_name}: {e}")
                                ollama_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": f"Error: Invalid arguments provided for tool {tool_name}"
                                })
                                continue

                            try:
                                logger.info(f"Calling tool {tool_name} on server {server_name}")
                                logger.debug(f"Tool arguments: {json.dumps(tool_arguments)}")
                                
                                # Log tool info before call
                                logger.debug(f"Tool details - Name: {tool_name}, Server: {server_name}")
                                
                                # Check if session is in mcp_sessions
                                if server_name not in self.mcp_sessions:
                                    raise Exception(f"Server {server_name} not found in active sessions")
                                
                                session = self.mcp_sessions[server_name]
                                
                                # Add timeout for tool call
                                async with asyncio.timeout(10):  # 10 seconds timeout for tool call
                                    mcp_tool_result = await session.call_tool(name=tool_name, arguments=tool_arguments)
                                    logger.debug(f"Raw tool result: {mcp_tool_result}")
                                    
                                    tool_result_content = mcp_tool_result.content
                                    logger.debug(f"Tool result content type: {type(tool_result_content)}")
                                    
                                    # Handle TextContent object
                                    if hasattr(tool_result_content, 'text'):
                                        logger.debug("Converting TextContent to text")
                                        tool_result_content = tool_result_content.text
                                    elif not isinstance(tool_result_content, (str, dict, list)):
                                        logger.debug(f"Converting {type(tool_result_content)} to string")
                                        tool_result_content = str(tool_result_content)
                                    
                                    # Only convert to JSON if it's a dict or list and not already a string
                                    if isinstance(tool_result_content, (dict, list)):
                                        logger.debug("Converting dict/list to JSON string")
                                        try:
                                            tool_result_content = json.dumps(tool_result_content)
                                        except TypeError as e:
                                            logger.error(f"Failed to serialize tool result: {e}")
                                            # Convert to string representation if JSON serialization fails
                                            tool_result_content = str(tool_result_content)
                                    elif not isinstance(tool_result_content, str):
                                        tool_result_content = str(tool_result_content)

                                    logger.info(f"Tool {tool_name} executed successfully on {server_name}")
                                    logger.debug(f"Final tool result content: {tool_result_content[:200]}...")
                                    
                                    # Ensure the content is not empty and is a string
                                    if not tool_result_content:
                                        logger.warning(f"Tool {tool_name} returned empty content")
                                        tool_result_content = f"The tool {tool_name} returned no results. Please try a different query."
                                    elif not isinstance(tool_result_content, str):
                                        tool_result_content = str(tool_result_content)
                                    
                                    ollama_messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_id,
                                        "name": tool_name,
                                        "content": tool_result_content
                                    })
                                    
                            except asyncio.TimeoutError:
                                error_msg = f"The {tool_name} tool timed out. Please try again."
                                logger.error(f"Timeout calling tool {tool_name} on {server_name}")
                                ollama_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": error_msg
                                })
                            except json.JSONDecodeError as je:
                                error_msg = f"Error processing {tool_name} results: Invalid JSON format"
                                logger.error(f"{error_msg}: {str(je)}")
                                ollama_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": error_msg
                                })
                            except Exception as e:
                                error_msg = f"The {tool_name} tool encountered an error. Please try again with a different query."
                                logger.error(f"Error calling tool {tool_name} on {server_name}: {str(e)}", exc_info=True)
                                ollama_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": error_msg
                                })

                        # Continue the loop to get LLM's response after tool calls
                        if "tools" in llm_kwargs:
                            del llm_kwargs["tools"]
                        if "tool_choice" in llm_kwargs:
                            del llm_kwargs["tool_choice"]
                        continue

                    else:  # No tool calls, LLM provided a direct response
                        full_response_content = assistant_message.content or ""
                        logger.info("LLM provided direct response (no tool calls).")
                        break

                except Exception as e:
                    logger.error(f"Error during LLM call in process_chat: {str(e)}", exc_info=True)
                    display_error(f"LLM API call failed: {str(e)}")
                    return

            # Handle max iterations reached
            if not full_response_content and current_iteration >= MAX_TOOL_CALL_ITERATIONS:
                logger.warning("Max tool call iterations reached without final response")
                full_response_content = "I seem to be stuck in a loop trying to use my tools. Could you please rephrase or try again?"

            # Process the final response
            if full_response_content:
                self.current_think_content = ""
                if "<think>" in full_response_content and "</think>" in full_response_content:
                    match = re.search(r"<think>(.*?)</think>", full_response_content, re.DOTALL)
                    if match:
                        self.current_think_content = match.group(1).strip()
                        full_response_content = re.sub(r"<think>.*?</think>", "", full_response_content, flags=re.DOTALL).strip()

            processed_response = await self._process_response(
                    full_response_content,
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

    async def _handle_sse_events(self, server_name: str, session: ClientSession):
        """Handle SSE events from a specific MCP server"""
        retry_count = 0
        max_retries = 3
        retry_delay = 5
        
        try:
            while True:
                if not session:
                    logger.warning(f"SSE connection lost for {server_name}")
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached for {server_name}, stopping event handler")
                        break
                    retry_count += 1
                    await asyncio.sleep(retry_delay)
                    continue

                try:
                    # Add timeout for event reading
                    async with asyncio.timeout(30):  # 30 seconds timeout for event reading
                        event = await session._reader.read()
                        if event:
                            event_type = event.get('event')
                            event_data = event.get('data')
                            
                            if event_type == 'heartbeat':
                                logger.debug(f"Received SSE heartbeat from {server_name}")
                                retry_count = 0  # Reset retry count on successful heartbeat
                            elif event_type == 'tool_update':
                                logger.info(f"Received tool update from {server_name}: {event_data}")
                                # Refresh tools for this server
                                tools_response = await session.list_tools()
                                if tools_response and tools_response.tools:
                                    # Remove old tools from this server
                                    self.mcp_tools = [tool for tool in self.mcp_tools 
                                                    if getattr(tool, 'server_name', None) != server_name]
                                    # Add updated tools
                                    for tool in tools_response.tools:
                                        tool.server_name = server_name
                                    self.mcp_tools.extend(tools_response.tools)
                                    logger.info(f"Updated tools for {server_name}")
                            elif event_type == 'error':
                                logger.error(f"Received SSE error from {server_name}: {event_data}")
                            else:
                                logger.debug(f"Received unknown SSE event type from {server_name}: {event_type}")
                                
                except asyncio.TimeoutError:
                    logger.debug(f"No events received from {server_name} in 30 seconds")
                    continue
                except Exception as e:
                    logger.error(f"Error processing SSE event from {server_name}: {str(e)}", exc_info=True)
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached for {server_name}, stopping event handler")
                        break
                    retry_count += 1
                    await asyncio.sleep(retry_delay)
                    continue
                
                await asyncio.sleep(1)  # Reduced from 0.1 to reduce log spam
                
        except asyncio.CancelledError:
            logger.info(f"SSE event handler task cancelled for {server_name}")
        except Exception as e:
            logger.error(f"SSE event handler encountered an error for {server_name}: {str(e)}", exc_info=True)

    async def cleanup(self):
        """Clean up resources used by the chat session"""
        if self.db:
            self.db.close()
            logger.info("Database session closed.")
        
        # Cancel all SSE tasks
        for server_name, task in self._sse_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"SSE event handler task cancelled for {server_name}")
        
        # Close all MCP sessions
        if self.mcp_sessions:
            await self._close_mcp_stack()

    async def _close_mcp_stack(self):
        """Close the MCP exit stack and all associated sessions"""
        logger.info("Closing MCP client resources...")
        await self.mcp_exit_stack.aclose()
        logger.info("MCP client resources closed.")

async def start_chat():
    """Main function to start and manage a chat session"""
    display_welcome()
    
    chat_session = ChatSession()
    try:
        # Initialize the session
        if not await chat_session.initialize():
            await chat_session.cleanup()
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
        await chat_session.cleanup() 