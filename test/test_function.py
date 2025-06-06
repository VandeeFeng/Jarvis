from openai import OpenAI
import os
import json
from datetime import datetime, timedelta

# Initialize the OpenAI client
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",  # Your local endpoint
    api_key="EMPTY"
)

# Define the tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format 'City, State, Country'."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit to return the temperature in. Defaults to 'celsius'."
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format 'City, State, Country'."
                    },
                    "date": {
                        "type": "string",
                        "description": "The date to get the temperature for, in the format 'YYYY-MM-DD'."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit to return the temperature in. Defaults to 'celsius'."
                    }
                },
                "required": ["location", "date"]
            }
        }
    }
]

def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location."""
    # This is a mock implementation - replace with actual API call
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit
    }

def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date."""
    # This is a mock implementation - replace with actual API call
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit
    }

def get_function_by_name(name):
    """Helper function to map function names to actual functions"""
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date
    return None

def get_weather_info(query: str):
    """Main function to handle weather queries using Qwen3"""
    
    # Set up the initial messages
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant. Current date: {datetime.now().strftime('%Y-%m-%d')}"
        },
        {
            "role": "user",
            "content": query
        }
    ]

    # Get the initial response from the model
    response = client.chat.completions.create(
        model="Qwen3:14b",  # Use appropriate model name
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Process the response
    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    # If no tool calls, return the response directly
    if not assistant_message.tool_calls:
        return assistant_message.content

    # Process each tool call
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Get and execute the function
        function = get_function_by_name(function_name)
        if function:
            function_response = function(**function_args)
            
            # Add the function response to messages
            messages.append({
                "role": "tool",
                "content": json.dumps(function_response),
                "tool_call_id": tool_call.id
            })

    # Get the final response from the model
    final_response = client.chat.completions.create(
        model="Qwen3:14b",  # Use appropriate model name
        messages=messages
    )

    return final_response.choices[0].message.content

if __name__ == "__main__":
    query = "What's the temperature in San Francisco now? How about tomorrow?"
    result = get_weather_info(query)
    print(result) 