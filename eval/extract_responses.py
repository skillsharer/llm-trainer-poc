#!/usr/bin/env python3
"""
Extract responses from artifacts for PromptFoo evaluation without exposing domain-specific content.
"""

import json
import os
import glob
from datetime import datetime


def extract_responses_from_artifacts():
    """
    Extract only the LLM responses from artifacts for evaluation.
    Keeps configuration generic while evaluating actual outputs.
    """
    # Get the script directory and find artifacts relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "..", "artifacts")
    output_file = os.path.join(script_dir, "data", "model_outputs.json")
    
    # Ensure data directory exists
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    responses = []
    
    # Get all artifact files
    json_files = glob.glob(os.path.join(artifacts_dir, "inference_results_*.json"))
    print(f"🔍 Found {len(json_files)} artifact files")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                artifact = json.load(f)
                results = artifact.get("results", [])
                print(f"📄 Processing {os.path.basename(json_file)}: {len(results)} results")
                
                for i, result in enumerate(results):
                    print(f"  - Result {i}: model={result.get('model', 'unknown')}, has_response={bool(result.get('response'))}")
                    
                    if result.get("response"):  # Simplified condition
                        response_data = {
                            "output": result.get("response", ""),
                            "model": result.get("model", "unknown"),
                            "metadata": {
                                "input_tokens": result.get("input_tokens", 0),
                                "output_tokens": result.get("output_tokens", 0),
                                "timestamp": artifact.get("timestamp", "unknown"),
                                "source_file": os.path.basename(json_file)
                            }
                        }
                        responses.append(response_data)
                        
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")
    
    # Save responses for PromptFoo in test case format
    test_cases = []
    for i, response in enumerate(responses):
        test_case = {
            "vars": {
                "response_to_evaluate": response["output"]
            },
            "metadata": response["metadata"]
        }
        test_cases.append(test_case)
    
    # Create PromptFoo test format
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    # Also save detailed metadata separately  
    metadata_file = os.path.join(script_dir, "data", "response_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Extracted {len(responses)} responses to {output_file}")
    print(f"📋 Use with: npx promptfoo eval -c promptfooconfig.yaml --model-outputs data/model_outputs.json")
    
    return output_file, len(responses)


if __name__ == "__main__":
    print("🎯 Extracting responses from artifacts...")
    output_file, count = extract_responses_from_artifacts()
    print(f"✅ Ready for evaluation with {count} responses!")
