# llm-trainer-poc

## Project setup
```bash
uv venv .venv
uv sync
```

## Token count for various foundation models
```bash
export $(grep -v '^#' .env | xargs)
```