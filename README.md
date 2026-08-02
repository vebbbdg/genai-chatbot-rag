# GenAI Chatbot | Full-Stack LLM Application with RAG

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built with FastAPI, LangChain, and DeepSeek LLM. Features real-time streaming responses, multi-session management, document upload (PDF/TXT/DOCX), and a modern ChatGPT-style web UI.

---

## ✨ Key Features

### 🤖 Core LLM Capabilities
- **Streaming Responses (SSE)** — Real-time token-by-token output, ChatGPT-like typing effect
- **Context Window Management** — Sliding window algorithm to prevent token overflow and optimize costs
- **Multi-Session Support** — Isolated conversation threads with auto-generated titles
- **System Prompt Engineering** — Configurable AI persona and behavior

### 📚 RAG (Retrieval-Augmented Generation)
- **Document Upload** — Support for PDF, TXT, and DOCX files
- **Intelligent Chunking** — Recursive character text splitting with overlap for context preservation
- **Vector Search** — ChromaDB vector store with semantic similarity retrieval
- **Local Embeddings** — BGE-small-en-v1.5 embedding model (runs offline, no API cost)
- **Toggle RAG Mode** — Switch between pure LLM and RAG-enhanced responses

### 🎨 Modern Web UI
- ChatGPT-inspired clean interface with sidebar navigation
- Markdown rendering with syntax-highlighted code blocks
- Responsive design for desktop and mobile
- Suggestion cards for quick start
- Document upload with RAG status indicator

### 🏗️ Engineering & Production
- **Layered Architecture** — Decoupled model, memory, session, and RAG modules
- **Structured Logging** — Dual output to console and rotating log files
- **CORS Support** — Ready for frontend-backend separation deployment
- **Environment Configuration** — Secure API key management via `.env`
- **Docker Support** — One-command deployment with Docker Compose
- **Health Check Endpoint** — Monitoring-ready `/api/health`

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)               │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Sidebar  │  │  Chat Area   │  │  Document Upload │   │
│  │ Sessions │  │  Markdown    │  │  RAG Toggle      │   │
│  └──────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          │ SSE Streaming (fetch + ReadableStream)
┌─────────────────────────▼───────────────────────────────┐
│                   FastAPI Backend                        │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ /api/chat  │  │ /api/sessions│ │ /api/documents   │  │
│  │  -stream   │  │ CRUD ops    │  │ Upload / Stats   │  │
│  └─────┬──────┘  └─────────────┘  └────────┬─────────┘  │
│        │                                    │            │
│  ┌─────▼──────────────────────────────────▼─────────┐   │
│  │              Core Business Logic                  │   │
│  │  ┌────────┐ ┌─────────┐ ┌───────┐ ┌──────────┐  │   │
│  │  │ Model  │ │ Memory  │ │Session│ │   RAG    │  │   │
│  │  │  LLM   │ │ Sliding │ │  Mgr  │ │ ChromaDB │  │   │
│  │  │Embed.  │ │ Window  │ │       │ │ Chunking │  │   │
│  │  └────────┘ └─────────┘ └───────┘ └──────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A DeepSeek API key (get one at [platform.deepseek.com](https://platform.deepseek.com))

### Local Development

```bash
# 1. Clone the repository
git clone <repo-url>
cd chat_robot

# 2. Create and activate virtual environment
conda create -n langchain1.2 python=3.11
conda activate langchain1.2

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY

# 5. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 6. Open in browser
# http://localhost:8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📁 Project Structure

```
chat_robot/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── core/
│   ├── __init__.py
│   ├── model.py          # LLM and embedding initialization
│   ├── memory.py         # Sliding window context management
│   ├── session.py        # Multi-session management
│   ├── rag.py            # RAG engine with ChromaDB
│   └── logger.py         # Structured logging
├── static/
│   └── index.html        # Single-page web application
├── data/
│   ├── vectordb/         # ChromaDB persistent storage
│   └── uploads/          # Uploaded documents
├── logs/                 # Application logs
└── tests/                # Unit tests
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat-stream` | Streaming chat (SSE) |
| `POST` | `/api/sessions` | Create new session |
| `GET` | `/api/sessions` | List all sessions |
| `DELETE` | `/api/sessions/{id}` | Delete session |
| `POST` | `/api/sessions/{id}/reset` | Reset session |
| `POST` | `/api/documents/upload` | Upload document |
| `GET` | `/api/documents/stats` | Knowledge base stats |
| `DELETE` | `/api/documents` | Clear knowledge base |

### Example: Streaming Chat
```bash
curl -X POST http://localhost:8000/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "use_rag": false}'
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Vanilla JS, Marked.js, Highlight.js, CSS3 |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **LLM** | LangChain, DeepSeek Chat API |
| **RAG** | ChromaDB, BGE Embeddings, PyPDF, python-docx |
| **Infra** | Docker, Python logging, CORS middleware |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=core --cov=main
```

---

## 🔮 Future Enhancements

- [ ] User authentication (JWT/OAuth)
- [ ] PostgreSQL/SQLite for persistent session storage
- [ ] Redis for caching and session management
- [ ] Multi-model support (OpenAI, Claude, local models)
- [ ] Streaming RAG with source citations
- [ ] WebSocket support for real-time features
- [ ] Rate limiting and API key management
- [ ] Deployment to AWS/GCP with CI/CD pipeline

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

Built as a portfolio project for AI Software Engineer applications.

**Tech Stack Keywords for Resume:**
`Python` `FastAPI` `LangChain` `RAG` `LLM` `ChromaDB` `Vector Search` `SSE Streaming`
`Full-Stack` `Docker` `REST API` `Pydantic` `Async` `Generative AI` `NLP`
