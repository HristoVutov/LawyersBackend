# Lawyers AI Backend

Python backend for the Lawyers AI legal assistant application.

## Quick Start

`ash
# Install dependencies
pip install -e .

# Run server
uvicorn app.main:app --reload --port 8000
`

## API Endpoints

- GET /api/health - Health check
- POST /api/project - Set active project path
- GET /api/project/mounts - Get virtual filesystem mounts
- WebSocket /ws/chat/{client_id} - Chat with agents

## Virtual File System

| Path | Description | Permissions |
|------|-------------|-------------|
| /templates/ | Global templates | Read/Write |
| /project/ | Current project | Read Only |
| /output/ | Generated files | Read/Write |

## Agents

- **orchestrator** - Routes tasks to specialized agents
- **document_agent** - File operations and document analysis
- **research_agent** - Legal research with TODO tracking
- **drafting_agent** - Document generation
- **template_agent** - Template management
- **analysis_agent** - Document analysis (JSON output)
