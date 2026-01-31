# Lawyers Analytics Service

Cloud analytics backend for collecting usage telemetry from Lawyers Dashboard.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your PostgreSQL connection string

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --port 8001
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/events` | Receive batch of telemetry events |
| `GET` | `/api/stats/daily` | Get daily aggregates |
| `GET` | `/health` | Health check |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `API_KEY` | Secret key for auth | Required |
| `CORS_ORIGINS` | Allowed origins | `*` |
