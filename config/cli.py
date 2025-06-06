from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text
from rich.console import Group
from rich.spinner import Spinner
from rich.style import Style
from rich.status import Status
from typing import List, Optional, AsyncGenerator, Dict
import asyncio
import re
import json

console = Console()

def display_welcome():
    """Display welcome message with styled commands."""
    welcome_panel = Panel.fit(
        """[bold green]Welcome to the Memory-Enhanced Chat System![/]
        
Available commands:
[blue]/search[/] [cyan]<keyword>[/] - Search memories by keyword
[blue]/similar[/] [cyan]<text>[/] - Search memories by semantic similarity
[blue]/exit[/] - Exit the chat""",
        title="🤖 Jarvis",
        border_style="bright_blue"
    )
    console.print(welcome_panel)
    console.print("\n[dim]Chat started...[/]")

def display_user_input(user_id: str):
    """Display user input prompt."""
    console.print(f"\n[bold blue]You ({user_id}):[/] ", end="")
    return console.input("")

def display_memories(memories: List, memory_type: str = ""):
    """Display found memories in a table."""
    if not memories:
        console.print(f"[yellow]No {memory_type} memories found.[/]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Content")
    table.add_column("Keyword", style="cyan")
    table.add_column("User", style="green")
    
    for mem in memories:
        table.add_row(
            mem.content,
            mem.keyword,
            mem.user_id or "N/A"
        )
    
    console.print(f"\n[bold]Found {len(memories)} {memory_type} memories:[/]")
    console.print(table)

async def display_streaming_content(
    content_stream: AsyncGenerator,
    prefix: str = "",
    is_thinking: bool = False,
    update_interval: float = 0.1
):
    """Display streaming content with optional thinking mode."""
    buffer = ""
    full_response = ""
    current_think_content = ""
    is_in_think_tag = False
    is_in_json = False
    json_brace_count = 0  # Track nested JSON braces
    last_panel_type = None  # Track the last panel type we displayed
    final_think_content = ""  # Store the final think content
    content_field_found = False  # Flag to track if we've found the "content": field
    memory_field_found = False  # Flag to track if we've found the "memory": field
    in_content_value = False  # Flag to track if we're currently in the content value
    in_memory_value = False  # Flag to track if we're currently in the memory value
    quote_count = 0  # Track quotes for field values
    
    def create_display_group():
        elements = []
        nonlocal last_panel_type, final_think_content
        
        if current_think_content.strip():
            elements.append(Panel(
                f"[yellow]{current_think_content.strip()}[/]",
                title=f"{prefix} Thinking...",
                border_style="yellow",
                padding=(0, 2)  # Reduce vertical padding
            ))
            last_panel_type = "think"
            final_think_content = current_think_content  # Save the current think content
        
        if buffer.strip():
            # Only add minimal spacing between panels
            content = buffer.strip()
            if last_panel_type == "think":
                # For assistant response after thinking, use a panel
                elements.append(Panel(
                    f"[bold green]{content}[/]",
                    title="🤖 Assistant",
                    border_style="green",
                    padding=(0, 2)
                ))
            else:
                # For continuous assistant response, just update the content
                elements.append(Text.from_markup(f"[bold green]{content}[/]"))
            last_panel_type = "message"
        
        # If we have elements, wrap them in a group with minimal spacing
        if elements:
            return Group(*elements)
        return Group(Text(""))

    def is_json_start(text: str) -> bool:
        """Check if the text appears to be the start of a JSON object."""
        cleaned = text.lstrip()
        return cleaned.startswith("{") and not cleaned.startswith("{think}")

    # Use a console with no extra spacing
    display_console = Console(force_terminal=True, no_color=False)
    
    with Live(
        create_display_group(),
        console=display_console,
        refresh_per_second=10,
        vertical_overflow="visible",
        auto_refresh=False,
        transient=True  # This helps reduce empty lines
    ) as live:
        temp_buffer_for_tag_detection = ""  # Temporary buffer for detecting start/end tags
        json_buffer = ""  # Buffer for detecting JSON content
        temp_content_buffer = ""  # Buffer for detecting fields

        async for chunk in content_stream:
            if chunk.choices[0].delta.content:
                chunk_content = chunk.choices[0].delta.content
                full_response += chunk_content
                
                # Process chunk char by char for robust detection
                for char in chunk_content:
                    # Track JSON braces
                    if char == "{":
                        json_buffer += char
                        if is_json_start(json_buffer) and not is_in_json:
                            is_in_json = True
                            json_brace_count = 1
                            content_field_found = False
                            memory_field_found = False
                            in_content_value = False
                            in_memory_value = False
                            quote_count = 0
                            json_buffer = ""
                            temp_content_buffer = ""
                            continue
                        elif is_in_json:
                            json_brace_count += 1
                            if in_content_value or in_memory_value:
                                buffer += char
                    elif char == "}" and is_in_json:
                        json_brace_count -= 1
                        if json_brace_count == 0:
                            is_in_json = False
                            json_buffer = ""
                            content_field_found = False
                            memory_field_found = False
                            in_content_value = False
                            in_memory_value = False
                            temp_content_buffer = ""
                        elif in_content_value or in_memory_value:
                            buffer += char
                        continue
                    elif is_in_json:
                        if not content_field_found and not memory_field_found:
                            temp_content_buffer += char
                            if '"content":' in temp_content_buffer:
                                content_field_found = True
                                temp_content_buffer = ""
                                in_content_value = True
                                quote_count = 0
                                continue
                            elif '"memory":' in temp_content_buffer:
                                memory_field_found = True
                                temp_content_buffer = ""
                                in_memory_value = True
                                quote_count = 0
                                continue
                        elif in_content_value or in_memory_value:
                            if char == '"':
                                quote_count += 1
                                if quote_count == 2:  # End of value
                                    if in_content_value:
                                        in_content_value = False
                                    else:
                                        in_memory_value = False
                                    continue
                            if in_content_value:  # Only add content to buffer
                                buffer += char
                        continue

                    if not is_in_json:
                        temp_buffer_for_tag_detection += char

                        if not is_in_think_tag:
                            if "<think>" in temp_buffer_for_tag_detection:
                                # Content before tag is message content
                                pre_tag_content = temp_buffer_for_tag_detection.split("<think>", 1)[0]
                                if pre_tag_content:
                                    buffer += pre_tag_content
                                is_in_think_tag = True
                                current_think_content = ""  # Reset think content
                                temp_buffer_for_tag_detection = ""  # Clear buffer after tag
                            elif len(temp_buffer_for_tag_detection) > 7:  # Max length of <think> or </think>
                                # If buffer is too long and no tag, flush it as message content
                                buffer += temp_buffer_for_tag_detection[0]
                                temp_buffer_for_tag_detection = temp_buffer_for_tag_detection[1:]

                        if is_in_think_tag:
                            if "</think>" in temp_buffer_for_tag_detection:
                                # Content before tag is think content
                                think_part = temp_buffer_for_tag_detection.split("</think>", 1)[0]
                                current_think_content += think_part
                                is_in_think_tag = False
                                temp_buffer_for_tag_detection = ""  # Clear buffer after tag
                            elif len(temp_buffer_for_tag_detection) > 8:  # Max length of </think>
                                # If buffer is too long and no tag, flush it as think content
                                current_think_content += temp_buffer_for_tag_detection[0]
                                temp_buffer_for_tag_detection = temp_buffer_for_tag_detection[1:]
                
                # After processing char_by_char, if anything remains in temp_buffer and not in JSON
                if temp_buffer_for_tag_detection and not is_in_json:
                    if is_in_think_tag:
                        current_think_content += temp_buffer_for_tag_detection
                    else:
                        buffer += temp_buffer_for_tag_detection
                    temp_buffer_for_tag_detection = ""
                    
                # Update display when we have any content to show
                if current_think_content or buffer.strip():
                    live.update(create_display_group())
                    live.refresh()

    # Final update to ensure everything is displayed
    live.update(create_display_group())
    live.refresh()

    # Return both the full response and the final think content
    return full_response, final_think_content

def clean_response_content(content: str) -> str:
    """Clean the response content by removing think tags and other unnecessary content."""
    if not content:
        return ""
    
    # Remove think content using regex
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # Remove any "response:" prefix
    content = re.sub(r'^Response:\s*', '', content, flags=re.MULTILINE).strip()
    return content

def display_assistant_response(response: str, fact: str = None, category: str = None, think_content: str = None):
    """Display the assistant's response with optional memory information and thinking process."""
    elements = []
    
    # Add thinking panel if we have thinking content
    if think_content and think_content.strip():
        elements.append(Panel(
            f"[yellow]{think_content.strip()}[/]",
            title="🤔 Thinking Process",
            border_style="yellow",
            padding=(0, 2)
        ))
    
    # Clean and add response panel
    cleaned_response = clean_response_content(response)
    if cleaned_response:
        # Remove any memory prefix if it exists
        if category and fact and cleaned_response.startswith(f"{category}: {fact}"):
            cleaned_response = cleaned_response[len(f"{category}: {fact}"):].strip()
            
        # Create a Markdown object for the response
        md = Markdown(cleaned_response)
        elements.append(Panel(
            md,
            title="🤖 Assistant",
            border_style="green",
            padding=(0, 2)
        ))
    
    # Add memory panel if fact exists
    if fact and category:
        elements.append(Panel.fit(
            f"[cyan]Stored new fact:[/] {fact}\n[cyan]Category:[/] {category}",
            title="📝 Memory",
            border_style="cyan",
            padding=(0, 2)
        ))
    
    # Display all panels in a group
    console.print(Group(*elements))

def display_error(error_msg: str):
    """Display error message."""
    error_panel = Panel(
        f"[bold red]{error_msg}[/]",
        title="❌ Error",
        border_style="red"
    )
    console.print(error_panel)

def display_tool_calling(tool_calls: List[Dict]):
    """Display tool calling information with rich formatting."""
    if not tool_calls:
        return

    # Create a table for tool calls
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title="🔧 Tool Calls",
        title_style="bold blue"
    )
    
    table.add_column("Tool Name", style="cyan")
    table.add_column("Arguments", style="green")
    
    for tool_call in tool_calls:
        # Handle both dict and OpenAI object formats
        if hasattr(tool_call, 'function'):
            # OpenAI format
            name = tool_call.function.name
            args = tool_call.function.arguments
        else:
            # Dict format
            name = tool_call.get("function", {}).get("name", "Unknown")
            args = tool_call.get("function", {}).get("arguments", "{}")
            
        # Format JSON arguments for better readability
        try:
            args_dict = json.loads(args)
            formatted_args = json.dumps(args_dict, indent=2)
        except:
            formatted_args = args
            
        table.add_row(name, formatted_args)

    # Create a panel with spinner
    spinner = Spinner("dots", style="blue")
    content = Group(
        Text("🔄 Processing tool calls...", style="bold blue"),
        table
    )
    
    panel = Panel.fit(
        content,
        title="Tool Execution",
        border_style="blue",
        padding=(0, 2)
    )
    
    console.print(panel)

def display_thinking():
    """Return a context manager for displaying thinking animation."""
    return Status("[bold green]🤔 Thinking...", spinner="aesthetic", spinner_style="green") 