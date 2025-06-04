# Jarvis

A from-scratch implementation of an AI agent with memory capabilities, No complex frameworks or black-box solutions - every component is custom-built for full control and understanding.

After experimenting with frameworks like LangChain, Mem0, and Agno, I found them somewhat cumbersome to build upon and extend. This project aims to provide a simpler, more straightforward implementation that you can easily understand and modify.

Most importantly, integrating LLMs for memory storage and management of important information makes this an excellent personal enhancement tool. It serves as your extended memory system, helping you manage and recall what matters most.

![pic1](/assets/pic1.png)

## Core Features

- 🧠 Custom-built memory system using raw vector operations in PostgreSQL
- 💡 Direct integration with AI models (OpenAI API / Ollama)
- 🔄 Hand-crafted context management and conversation handling
- 📌 Ground-up implementation of persistent memory
- 🎯 Custom tracking system for tasks and conversation history

## Architecture

- Raw vector operations for memory storage and retrieval
- Pure Python async implementation
- Built from first principles with minimal dependencies
- Custom logging and monitoring system

## Tech Stack

- Pure Python 3.x
- PostgreSQL + pgvector (raw SQL operations)
- Direct API integration with AI models
- Minimal FastAPI setup for async support
- Basic SQLAlchemy for DB operations

## Quick Start

First, start the database:
```bash
cd db
docker compose up -d  # This will start PostgreSQL with pgvector extension
```

Then setup the project:
```bash
pip install -r requirements.txt
cp .env.example .env  # Configure your AI API keys and DB settings
python jarvis.py
``` 