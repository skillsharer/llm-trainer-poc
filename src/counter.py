import os
import tiktoken
import anthropic
import google.generativeai as genai
from transformers import AutoTokenizer
from modules.data_loader import load_json, load_pdf

class TokenCounter:
    """
    A class to count tokens for various LLMs.
    Initializes all necessary tokenizers once to be efficient.
    """
    def __init__(self):
        print("Initializing tokenizers... ⏳")
        self.hf_tokenizers = {}
        hf_models = {
            "llama": "meta-llama/Llama-3.1-8B",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "mixtral": "mistralai/Mixtral-8x7B-v0.1",
            "qwen": "Qwen/Qwen1.5-7B-Chat",
        }
        for name, path in hf_models.items():
            try:
                self.hf_tokenizers[name] = AutoTokenizer.from_pretrained(path)
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {name} ({path}). Error: {e}")

        # --- OpenAI Tokenizer (Tiktoken) ---
        # Models like gpt-4, gpt-3.5-turbo use cl100k_base
        try:
            self.openai_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Warning: Could not load tiktoken encoder. Error: {e}")
            self.openai_encoding = None
            
        # --- Anthropic Tokenizer ---
        try:
            self.anthropic_client = anthropic.Anthropic()
        except Exception as e:
            print(f"Warning: Could not load Anthropic client. Error: {e}")
            self.anthropic_client = None

        # --- Google Gemini (API-based) ---
        # The official way to count Gemini tokens is via the API.
        # Ensure your GOOGLE_API_KEY is set as an environment variable.
        try:
            if "GOOGLE_API_KEY" in os.environ:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                print("Gemini tokenizer configured successfully. ✅")
            else:
                self.gemini_model = None
                print("Warning: GOOGLE_API_KEY not found. Gemini token counting will be skipped.")
        except Exception as e:
            print(f"Warning: Could not configure Gemini. Error: {e}")
            self.gemini_model = None
        
        print("Tokenizers initialized. Ready to count. 👍")

    def count(self, model_family: str, text: str) -> int:
        """
        Counts tokens for a given model family and text.

        Args:
            model_family (str): One of 'openai', 'gemini', 'anthropic', 'llama', 'mistral', etc.
            text (str): The input text to count tokens for.

        Returns:
            int: The number of tokens, or -1 if the tokenizer is not available.
        """
        model_family = model_family.lower()

        if model_family in ["openai", "gpt-4", "gpt-3.5-turbo"]:
            if self.openai_encoding:
                return len(self.openai_encoding.encode(text))
        elif model_family in ["gemini", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]:
            if self.gemini_model:
                # This makes a lightweight API call for an accurate count.
                return self.gemini_model.count_tokens(text).total_tokens
        elif model_family in ["anthropic"]:
            # https://medium.com/data-science/introducing-the-new-anthropic-token-counting-api-5afd58bad5ff
            if self.anthropic_client and "ANTHROPIC_API_KEY" in os.environ:
                # Use the new beta token counting API
                response = self.anthropic_client.beta.messages.count_tokens(
                    betas=["token-counting-2024-11-01"],
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": text}]
                )
                return response.input_tokens
        elif model_family in self.hf_tokenizers:
            tokenizer = self.hf_tokenizers[model_family]
            return len(tokenizer.encode(text))
        
        return -1

def main():
    data_dir = os.environ["DATA_DIR"]
    input_text = ""
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if file_name.endswith(".pdf"):
            input_text += load_pdf(file_path)
        elif file_name.endswith(".json"):
            input_text += load_json(file_path)
        else:
            continue
    counter = TokenCounter()

    models_to_test = [
        "openai", 
        "gemini", 
        "anthropic", 
        "llama", 
        "qwen", 
        "mistral", 
        "mixtral"
    ]

    print("\n" + "="*40)
    print("📏 Token Counts for Input Text:")
    print(f"Input Text Length: {len(input_text)} characters")
    print("="*40)

    # 3. Loop and count
    for model in models_to_test:
        try:
            token_count = counter.count(model, input_text)
            if token_count != -1:
                # The f-string formatting aligns the output neatly
                print(f"🔹 {model.capitalize():<12} | {token_count} tokens")
            else:
                print(f"🔸 {model.capitalize():<12} | Tokenizer not available.")
        except Exception as e:
            print(f"❌ {model.capitalize():<12} | Failed to count tokens. Error: {e}")
    
    print("\n✅ Done.")


if __name__ == "__main__":
    main()