#!/usr/bin/env python3
"""
Setup PromptFoo evaluation environment for LLM Trainer POC
"""

import json
import os
import glob
import yaml
from datetime import datetime


def setup_promptfoo_evaluation():
    """
    Setup the complete PromptFoo evaluation environment.
    """
    print("🎯 Setting up PromptFoo Evaluation Environment...")
    
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(current_dir, "..", "artifacts")
    data_dir = os.path.join(current_dir, "data")
    results_dir = os.path.join(current_dir, "results")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 Created directories: {data_dir}, {results_dir}")
    
    # Load artifacts to understand the data structure
    json_files = glob.glob(os.path.join(artifacts_dir, "inference_results_*.json"))
    print(f"🔍 Found {len(json_files)} artifact files")
    
    if not json_files:
        print("❌ No inference artifacts found. Please run inference.py first.")
        return False
    
    # Analyze existing results
    all_models = set()
    total_test_cases = 0
    sample_inputs = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                artifact = json.load(f)
                results = artifact.get("results", [])
                
                for result in results:
                    if "error" not in result:
                        all_models.add(result.get("model", "unknown"))
                        total_test_cases += 1
                        
                        # Extract input text from prompt
                        if "prompt" in result:
                            prompt = result["prompt"]
                            if "Please summarize the following text:" in prompt:
                                input_text = prompt.split("Please summarize the following text:")[-1].strip()
                                if input_text.startswith("\n\n"):
                                    input_text = input_text[2:]
                                if input_text and len(input_text) > 50:  # Only meaningful inputs
                                    sample_inputs.append(input_text)
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    print(f"📊 Found {total_test_cases} test cases from models: {', '.join(all_models)}")
    print(f"📝 Extracted {len(sample_inputs)} sample inputs")
    
    if not sample_inputs:
        print("❌ No valid sample inputs found in artifacts.")
        return False
    
    # Verify existing config file
    config_file = os.path.join(current_dir, "promptfooconfig.yaml")
    if os.path.exists(config_file):
        print(f"⚙️  Using existing config: {config_file}")
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config loaded successfully - general evaluation ready")
        except Exception as e:
            print(f"⚠️  Warning: Could not validate config file: {e}")
    else:
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Create analysis of existing results for comparison
    analysis = {
        "evaluation_type": "setup_analysis",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_artifacts": len(json_files),
            "total_test_cases": total_test_cases,
            "models_found": list(all_models)
        },
        "sample_responses": []
    }
    
    # Collect sample responses for comparison
    for json_file in json_files[:2]:  # Just first 2 files for samples
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                artifact = json.load(f)
                results = artifact.get("results", [])
                
                for result in results:
                    if "error" not in result and "response" in result:
                        analysis["sample_responses"].append({
                            "model": result.get("model", "unknown"),
                            "response_preview": result.get("response", "")[:300] + "..." if len(result.get("response", "")) > 300 else result.get("response", ""),
                            "input_tokens": result.get("input_tokens", 0),
                            "output_tokens": result.get("output_tokens", 0),
                            "timestamp": artifact.get("timestamp", "unknown")
                        })
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")
    
    # Save analysis
    analysis_file = os.path.join(data_dir, "setup_analysis.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Created setup analysis: {analysis_file}")
    
    return True

def main():
    """
    Main setup function.
    """
    print("🎯 PromptFoo Evaluation Setup")
    print("=" * 40)
    
    # Run setup
    success = setup_promptfoo_evaluation()
    
    if success:
        print("✅ Setup completed successfully!")
    else:
        print("❌ Setup failed. Please check artifacts directory.")


if __name__ == "__main__":
    main()
