# llm-trainer-poc

## Project setup
1. Install `uv` for the project from [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

2. Setup `.venv`
```bash
uv venv .venv
uv sync
```

3. Obtain Huggingface User Access Token. See description [here](https://huggingface.co/docs/hub/en/security-tokens).

4. Obtain Google Gemini API key [here](https://aistudio.google.com/app/apikey).

5. Possible access requests needed additionally in hunggingface for local models.

## Token count for various foundation models


```bash
export $(grep -v '^#' .env | xargs)
uv run ./src/counter.py
```