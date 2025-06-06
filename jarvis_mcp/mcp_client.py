import json
import logging
import asyncio
from contextlib import AsyncExitStack
from typing import Dict, List, Optional, Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from config.config import MCP_SERVER

logger = logging.getLogger(__name__)

class MCPToolsManager:
    def __init__(self):
        self.mcp_config = MCP_SERVER
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.mcp_exit_stack = AsyncExitStack()
        self.mcp_tools: List[Any] = []
        self._sse_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> bool:
        """Initialize MCP connections and tools"""
        if not self.mcp_config.get("enabled"):
            logger.info("MCP client is disabled.")
            return False

        try:
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
                            async with asyncio.timeout(10):
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
                        continue
                
                if not self.mcp_sessions:
                    logger.error("Failed to connect to any MCP servers")
                    return False
                else:
                    logger.info(f"Successfully connected to {len(self.mcp_sessions)} MCP servers")
                    logger.info(f"Total available tools: {len(self.mcp_tools)}")
                    return True
            else:
                logger.error(f"Unsupported MCP server type: {server_type}")
                return False

        except Exception as e:
            logger.error(f"Error during MCP initialization: {str(e)}", exc_info=True)
            return False

    async def _handle_sse_events(self, server_name: str, session: ClientSession):
        """Handle SSE events from a specific MCP server"""
        retry_count = 0
        max_retries = 3
        retry_delay = 5
        
        try:
            while True:
                try:
                    # Add timeout for event reading
                    async with asyncio.timeout(30):  # 30 seconds timeout for event reading
                        # Use the _read_stream to get events
                        event = await session._read_stream.read()
                        if event:
                            try:
                                event_data = json.loads(event.decode())
                                event_type = event_data.get('type')
                                
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
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse SSE event from {server_name}: {e}")
                                continue
                                    
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

    def get_tools(self) -> List[Any]:
        """Get all available MCP tools"""
        return self.mcp_tools

    def get_session(self, server_name: str) -> Optional[ClientSession]:
        """Get a specific MCP session by server name"""
        return self.mcp_sessions.get(server_name)

    def format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Format available MCP tools for LLM consumption."""
        formatted_mcp_tools = []
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
        return formatted_mcp_tools

    async def call_tool(self, tool_name: str, tool_arguments: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Call an MCP tool and return its result"""
        # Find the tool and its server
        tool = next((t for t in self.mcp_tools if t.name == tool_name), None)
        if not tool:
            error_msg = f"Tool {tool_name} not found in any MCP server"
            logger.error(error_msg)
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": f"Error: {error_msg}"
            }

        server_name = getattr(tool, 'server_name', None)
        if not server_name or server_name not in self.mcp_sessions:
            error_msg = f"Server not found for tool {tool_name}"
            logger.error(error_msg)
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": f"Error: {error_msg}"
            }

        session = self.mcp_sessions[server_name]
        try:
            logger.info(f"Calling tool {tool_name} on server {server_name}")
            logger.debug(f"Tool arguments: {json.dumps(tool_arguments)}")
            
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
                
                return {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": tool_result_content
                }
                
        except asyncio.TimeoutError:
            error_msg = f"The {tool_name} tool timed out. Please try again."
            logger.error(f"Timeout calling tool {tool_name} on {server_name}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": error_msg
            }
        except json.JSONDecodeError as je:
            error_msg = f"Error processing {tool_name} results: Invalid JSON format"
            logger.error(f"{error_msg}: {str(je)}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": error_msg
            }
        except Exception as e:
            error_msg = f"The {tool_name} tool encountered an error. Please try again with a different query."
            logger.error(f"Error calling tool {tool_name} on {server_name}: {str(e)}", exc_info=True)
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": error_msg
            }

    async def cleanup(self):
        """Clean up MCP resources"""
        # Cancel all SSE tasks
        for server_name, task in self._sse_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"SSE event handler task cancelled for {server_name}")
        
        # Close all MCP sessions
        await self._close_mcp_stack()

    async def _close_mcp_stack(self):
        """Close all MCP sessions and SSE tasks"""
        for task in self._sse_tasks.values():
            task.cancel()
        await asyncio.gather(*self._sse_tasks.values(), return_exceptions=True)
        self._sse_tasks.clear()
        await self.mcp_exit_stack.aclose()
        logger.info("MCP sessions and SSE tasks closed.") 