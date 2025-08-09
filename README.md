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

5. Obtain Claude API key [here](https://console.anthropic.com/settings/keys).

6. Obtain OpenAI API key [here](https://platform.openai.com/api-keys).

7. Use `./setup-env.sh` in the `src` directory then add the created API keys to the `.env`

8. Possible access requests needed additionally in hunggingface for local models.

9. Add your `.pdf` and `.json` files to the `<repo_root>/data` directory. 

## Token count for various foundation models


```bash
export $(grep -v '^#' ./src/.env | xargs)
uv run ./src/counter.py
```

## Inference for various foundation models

```bash
export $(grep -v '^#' ./src/.env | xargs)
uv run ./src/inference.py
```