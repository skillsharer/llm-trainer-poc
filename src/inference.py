import os
import json
import datetime
import google.generativeai as genai
from transformers import AutoModelForCausalLM
import torch
from openai import OpenAI
import anthropic
from modules.data_loader import load_json, load_pdf
from counter import TokenCounter


class LLMInference:
    """
    A class to perform inference with various LLMs and track token usage.
    Uses the existing TokenCounter class for consistent token counting.
    """
    
    def __init__(self):
        print("Initializing LLM inference clients... ⏳")
        
        # Initialize the TokenCounter for consistent token counting
        self.token_counter = TokenCounter()
        
        # Store model references for inference
        self.hf_models = {}
        hf_model_configs = {
            "llama": "meta-llama/Llama-3.1-8B",
            "mistral": "mistralai/Mistral-7B-v0.1", 
            "mixtral": "mistralai/Mixtral-8x7B-v0.1",
            "qwen": "Qwen/Qwen1.5-7B-Chat",
        }
        
        # Load Hugging Face models for inference (reuse tokenizers from TokenCounter)
        for name, path in hf_model_configs.items():
            try:
                if name in self.token_counter.hf_tokenizers and self.token_counter.hf_tokenizers[name]:
                    # Only load models if we have GPU available
                    if torch.cuda.is_available():
                        print(f"Loading {name} model on GPU...")
                        self.hf_models[name] = AutoModelForCausalLM.from_pretrained(
                            path, 
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )
                    else:
                        print(f"Warning: No GPU available for {name}. Skipping model loading.")
                        self.hf_models[name] = None
                else:
                    print(f"Warning: Tokenizer for {name} not available. Skipping model loading.")
                    self.hf_models[name] = None
            except Exception as e:
                print(f"Warning: Could not load {name} model. Error: {e}")
                self.hf_models[name] = None

        # Initialize OpenAI client
        try:
            self.openai_client = OpenAI() if "OPENAI_API_KEY" in os.environ else None
            if self.openai_client:
                print("OpenAI client configured successfully. ✅")
        except Exception as e:
            print(f"Warning: Could not load OpenAI client. Error: {e}")
            self.openai_client = None
            
        # Initialize Anthropic client
        try:
            self.anthropic_client = anthropic.Anthropic() if "ANTHROPIC_API_KEY" in os.environ else None
            if self.anthropic_client:
                print("Anthropic client configured successfully. ✅")
        except Exception as e:
            print(f"Warning: Could not load Anthropic client. Error: {e}")
            self.anthropic_client = None

        # Initialize Google Gemini
        try:
            if "GOOGLE_API_KEY" in os.environ:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                print("Gemini client configured successfully. ✅")
            else:
                self.gemini_model = None
                print("Warning: GOOGLE_API_KEY not found. Gemini inference will be skipped.")
        except Exception as e:
            print(f"Warning: Could not configure Gemini. Error: {e}")
            self.gemini_model = None
        
        print("LLM inference setup completed. 👍")

    def count_tokens(self, model_family: str, text: str) -> int:
        """
        Use the existing TokenCounter for consistent token counting.
        """
        return self.token_counter.count(model_family, text)

    def infer_openai(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1000) -> dict:
        """Perform inference using OpenAI models"""
        if not self.openai_client:
            return {"error": "OpenAI client not available"}
        
        try:
            input_tokens = self.count_tokens("openai", prompt)
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            output_text = response.choices[0].message.content
            output_tokens = self.count_tokens("openai", output_text)
            
            return {
                "model": model,
                "response": output_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        except Exception as e:
            return {"error": f"OpenAI inference failed: {e}"}

    def infer_anthropic(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 1000) -> dict:
        """Perform inference using Anthropic models"""
        if not self.anthropic_client:
            return {"error": "Anthropic client not available"}
        
        try:
            input_tokens = self.count_tokens("anthropic", prompt)
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            output_text = response.content[0].text
            output_tokens = self.count_tokens("anthropic", output_text)
            
            return {
                "model": model,
                "response": output_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        except Exception as e:
            return {"error": f"Anthropic inference failed: {e}"}

    def infer_gemini(self, prompt: str, model: str = "gemini-1.5-flash", max_tokens: int = 1000) -> dict:
        """Perform inference using Google Gemini models"""
        if not self.gemini_model:
            return {"error": "Gemini client not available"}
        
        try:
            input_tokens = self.count_tokens("gemini", prompt)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens
            )
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            output_text = response.text
            output_tokens = self.count_tokens("gemini", output_text)
            
            return {
                "model": model,
                "response": output_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        except Exception as e:
            return {"error": f"Gemini inference failed: {e}"}

    def infer_huggingface(self, prompt: str, model_family: str, max_tokens: int = 1000) -> dict:
        """Perform inference using Hugging Face models"""
        if model_family not in self.hf_models or not self.hf_models[model_family]:
            return {"error": f"Hugging Face model {model_family} not available"}
        
        try:
            model = self.hf_models[model_family]
            tokenizer = self.token_counter.hf_tokenizers[model_family]  # Use tokenizer from TokenCounter
            
            # Tokenize input
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            input_tokens = len(inputs[0])
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response (exclude input tokens)
            response_tokens = outputs[0][len(inputs[0]):]
            output_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            output_tokens = len(response_tokens)
            
            return {
                "model": model_family,
                "response": output_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        except Exception as e:
            return {"error": f"Hugging Face inference failed: {e}"}

    def infer(self, model_family: str, prompt: str, max_tokens: int = 1000) -> dict:
        """
        Perform inference with the specified model family
        
        Args:
            model_family (str): One of 'openai', 'anthropic', 'gemini', 'llama', 'mistral', etc.
            prompt (str): The input prompt
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            dict: Response containing model, response text, and token counts
        """
        model_family = model_family.lower()
        
        if model_family in ["openai", "gpt-4", "gpt-3.5-turbo"]:
            return self.infer_openai(prompt, max_tokens=max_tokens)
        elif model_family in ["anthropic"]:
            return self.infer_anthropic(prompt, max_tokens=max_tokens)
        elif model_family in ["gemini", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]:
            return self.infer_gemini(prompt, max_tokens=max_tokens)
        elif model_family in ["llama", "mistral", "mixtral", "qwen"]:
            return self.infer_huggingface(prompt, model_family, max_tokens=max_tokens)
        else:
            return {"error": f"Unsupported model family: {model_family}"}


def save_results_to_json(results: list, output_file: str):
    """Save inference results to a JSON file"""
    timestamp = datetime.datetime.now().isoformat()
    output_data = {
        "timestamp": timestamp,
        "results": results
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    # Load input text the same way as counter.py
    data_dir = os.environ.get("DATA_DIR", "data")
    artifacsts_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    input_text = ""
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if file_name.endswith(".pdf"):
            input_text += load_pdf(file_path)
        elif file_name.endswith(".json"):
            input_text += load_json(file_path)
        else:
            continue
    
    if not input_text.strip():
        print("❌ No input text loaded from data directory")
        return
    
    # Truncate input text if too long (to manage costs and processing time)
    max_input_length = os.environ.get("INPUT_TOKEN_LIMIT", 4000)
    if len(input_text) > max_input_length:
        input_text = input_text[:max_input_length] + "\n[... truncated ...]"
        print(f"⚠️  Input text truncated to {max_input_length} characters")
    
    # Initialize inference engine
    inference_engine = LLMInference()
    
    # Models to test (same as counter.py)
    models_to_test = [
        "openai", 
        "gemini", 
        "anthropic", 
        #"llama", 
        #"qwen", 
        #"mistral", 
        #"mixtral"
    ]
    
    prompt = f"Please summarize the following text:\n\n{input_text}"
    
    print("\n" + "="*60)
    print("🤖 LLM Inference Results:")
    print(f"Input Text Length: {len(input_text)} characters")
    print("="*60)
    
    results = []
    
    # Perform inference with each model
    for model in models_to_test:
        print(f"\n🔹 Testing {model.capitalize()}...")
        try:
            result = inference_engine.infer(model, prompt, max_tokens=500)
            
            if "error" in result:
                print(f"❌ {model.capitalize():<12} | {result['error']}")
                result["model_family"] = model
                result["prompt"] = prompt
            else:
                print(f"✅ {model.capitalize():<12} | Generated {result['output_tokens']} tokens")
                print(f"   Input: {result['input_tokens']} tokens, Output: {result['output_tokens']} tokens, Total: {result['total_tokens']} tokens")
                print(f"   Response: {result['response'][:100]}..." if len(result['response']) > 100 else f"   Response: {result['response']}")
                result["model_family"] = model
                result["prompt"] = prompt
                
            results.append(result)
            
        except Exception as e:
            error_result = {
                "model_family": model,
                "prompt": prompt,
                "error": f"Unexpected error: {e}"
            }
            results.append(error_result)
            print(f"❌ {model.capitalize():<12} | Unexpected error: {e}")
    
    # Save results to JSON file
    output_file = os.path.join(artifacsts_dir, f"inference_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results_to_json(results, output_file)
    
    print("\n" + "="*60)
    print("📊 Summary:")
    successful_inferences = [r for r in results if "error" not in r]
    print(f"✅ Successful inferences: {len(successful_inferences)}/{len(results)}")
    
    if successful_inferences:
        total_input_tokens = sum(r.get("input_tokens", 0) for r in successful_inferences)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in successful_inferences)
        print(f"📈 Total input tokens: {total_input_tokens}")
        print(f"📈 Total output tokens: {total_output_tokens}")
        print(f"📈 Total tokens: {total_input_tokens + total_output_tokens}")
    
    print(f"💾 Results saved to: {output_file}")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()