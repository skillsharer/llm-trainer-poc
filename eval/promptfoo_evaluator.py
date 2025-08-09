#!/usr/bin/env python3
"""
PromptFoo integration for LLM inference results.
This module converts inference artifacts to promptfoo format and runs evaluations.
"""

import json
import os
import glob
import subprocess
from datetime import datetime
from typing import Dict, List, Any


class PromptFooEvaluator:
    """
    Integrates existing LLM inference results with PromptFoo evaluation framework.
    """
    
    def __init__(self, artifacts_dir: str = "../artifacts", eval_dir: str = "."):
        """
        Initialize the PromptFoo evaluator.
        
        Args:
            artifacts_dir: Directory containing inference result JSON files (relative to eval dir)
            eval_dir: Evaluation directory (current directory by default)
        """
        self.artifacts_dir = artifacts_dir
        self.eval_dir = eval_dir
        self.data_dir = os.path.join(eval_dir, "data")
        self.results_dir = os.path.join(eval_dir, "results")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_inference_artifacts(self) -> List[Dict[str, Any]]:
        """
        Load all inference result JSON files from the artifacts directory.
        
        Returns:
            List of parsed JSON artifacts
        """
        artifacts = []
        
        # Get all JSON files in artifacts directory
        json_files = glob.glob(os.path.join(self.artifacts_dir, "inference_results_*.json"))
        
        for file_path in sorted(json_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artifact = json.load(f)
                    artifact['file_path'] = file_path
                    artifact['file_name'] = os.path.basename(file_path)
                    artifacts.append(artifact)
                    print(f"✅ Loaded artifact: {artifact['file_name']}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        return artifacts
    
    def extract_test_cases(self, artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract test cases from inference artifacts for promptfoo evaluation.
        
        Args:
            artifacts: List of inference artifacts
            
        Returns:
            List of test cases in promptfoo format
        """
        test_cases = []
        
        for artifact in artifacts:
            results = artifact.get("results", [])
            
            for result in results:
                if "error" not in result and "prompt" in result:
                    # Extract the input data from the prompt
                    prompt = result["prompt"]
                    
                    # Try to extract the actual input text after "Please summarize the following text:"
                    if "Please summarize the following text:" in prompt:
                        input_data = prompt.split("Please summarize the following text:")[-1].strip()
                        if input_data.startswith("\n\n"):
                            input_data = input_data[2:]
                    else:
                        input_data = prompt
                    
                    test_case = {
                        "vars": {
                            "scenario_data": input_data
                        },
                        "expected_response": result.get("response", ""),
                        "model": result.get("model", "unknown"),
                        "model_family": result.get("model_family", "unknown"),
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                        "artifact_timestamp": artifact.get("timestamp", "unknown")
                    }
                    
                    test_cases.append(test_case)
        
        return test_cases
    
    def create_promptfoo_dataset(self, test_cases: List[Dict[str, Any]]) -> str:
        """
        Create a promptfoo dataset file from test cases.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Path to the created dataset file
        """
        dataset_file = os.path.join(self.data_dir, "dataset.json")
        
        # Group test cases by unique input to avoid duplicates
        unique_inputs = {}
        for test_case in test_cases:
            input_data = test_case["vars"]["scenario_data"]
            input_hash = hash(input_data)
            
            if input_hash not in unique_inputs:
                unique_inputs[input_hash] = {
                    "vars": test_case["vars"],
                    "models_tested": [],
                    "responses": {}
                }
            
            unique_inputs[input_hash]["models_tested"].append({
                "model": test_case["model"],
                "model_family": test_case["model_family"],
                "response": test_case["expected_response"],
                "input_tokens": test_case["input_tokens"],
                "output_tokens": test_case["output_tokens"],
                "timestamp": test_case["artifact_timestamp"]
            })
        
        # Convert to promptfoo dataset format
        dataset = []
        for input_hash, data in unique_inputs.items():
            dataset.append({
                "vars": data["vars"],
                "metadata": {
                    "models_tested": data["models_tested"],
                    "input_hash": input_hash
                }
            })
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Created promptfoo dataset: {dataset_file} ({len(dataset)} test cases)")
        return dataset_file
    
    def generate_comparison_report(self, test_cases: List[Dict[str, Any]]) -> str:
        """
        Generate a comparison report from existing model responses.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Path to the comparison report file
        """
        # Group responses by input for comparison
        input_groups = {}
        for test_case in test_cases:
            input_data = test_case["vars"]["scenario_data"]
            input_hash = hash(input_data)
            
            if input_hash not in input_groups:
                input_groups[input_hash] = {
                    "input": input_data[:200] + "..." if len(input_data) > 200 else input_data,
                    "responses": []
                }
            
            input_groups[input_hash]["responses"].append({
                "model": test_case["model"],
                "response": test_case["expected_response"],
                "input_tokens": test_case["input_tokens"],
                "output_tokens": test_case["output_tokens"]
            })
        
        # Create comparison results file
        comparison_data = {
            "evaluation_type": "artifact_comparison",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_inputs": len(input_groups),
                "total_responses": len(test_cases)
            },
            "comparisons": []
        }
        
        for input_hash, data in input_groups.items():
            if len(data["responses"]) > 1:  # Only include inputs with multiple model responses
                comparison = {
                    "input_preview": data["input"],
                    "input_hash": input_hash,
                    "models_compared": len(data["responses"]),
                    "responses": data["responses"]
                }
                comparison_data["comparisons"].append(comparison)
        
        comparison_file = os.path.join(self.data_dir, "comparison_results.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"🔄 Created comparison results: {comparison_file}")
        return comparison_file
    
    def run_promptfoo_eval(self, config_file: str = "promptfooconfig.yaml") -> str:
        """
        Run promptfoo evaluation using the configuration.
        
        Args:
            config_file: Path to the promptfoo config file
            
        Returns:
            Path to the evaluation results
        """
        try:
            # Check if promptfoo is available
            result = subprocess.run(
                ["npx", "promptfoo", "--version"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.eval_dir
            )
            print(f"✅ PromptFoo version: {result.stdout.strip()}")
            
            # Run evaluation
            print("🚀 Running promptfoo evaluation...")
            eval_command = [
                "npx", "promptfoo", "eval",
                "-c", config_file,
                "-o", os.path.join(self.results_dir, "eval_results.json")
            ]
            
            result = subprocess.run(
                eval_command,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.eval_dir
            )
            
            print("✅ PromptFoo evaluation completed!")
            print(f"Output: {result.stdout}")
            
            return self.results_dir
            
        except subprocess.CalledProcessError as e:
            print(f"❌ PromptFoo evaluation failed: {e}")
            print(f"Error output: {e.stderr}")
            return None
        except FileNotFoundError:
            print("❌ PromptFoo not found. Make sure it's installed with 'npm install promptfoo'")
            return None
    
    def view_results(self, results_dir: str = None):
        """
        Open promptfoo results viewer.
        
        Args:
            results_dir: Directory containing promptfoo results
        """
        if results_dir is None:
            results_dir = self.results_dir
            
        try:
            print("🌐 Opening promptfoo results viewer...")
            subprocess.run(
                ["npx", "promptfoo", "view", results_dir],
                check=True,
                cwd=self.eval_dir
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to open results viewer: {e}")
        except FileNotFoundError:
            print("❌ PromptFoo not found. Make sure it's installed with 'npm install promptfoo'")
    
    def generate_summary_report(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report from the test cases.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Summary report dictionary
        """
        # Analyze model performance
        model_stats = {}
        
        for test_case in test_cases:
            model = test_case["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "response_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "responses": []
                }
            
            stats = model_stats[model]
            stats["response_count"] += 1
            stats["total_input_tokens"] += test_case["input_tokens"]
            stats["total_output_tokens"] += test_case["output_tokens"]
            stats["responses"].append(len(test_case["expected_response"]))
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["response_count"] > 0:
                stats["avg_input_tokens"] = stats["total_input_tokens"] / stats["response_count"]
                stats["avg_output_tokens"] = stats["total_output_tokens"] / stats["response_count"]
                stats["avg_response_length"] = sum(stats["responses"]) / len(stats["responses"])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "models_analyzed": len(model_stats),
            "model_statistics": model_stats,
            "recommendations": self._generate_recommendations(model_stats)
        }
        
        return summary
    
    def _generate_recommendations(self, model_stats: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on model statistics.
        
        Args:
            model_stats: Dictionary of model statistics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not model_stats:
            return ["No model statistics available for recommendations."]
        
        # Find most efficient model (best output/input token ratio)
        most_efficient = None
        best_efficiency = 0
        
        for model, stats in model_stats.items():
            if stats["avg_input_tokens"] > 0:
                efficiency = stats["avg_output_tokens"] / stats["avg_input_tokens"]
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    most_efficient = model
        
        if most_efficient:
            recommendations.append(f"Most token-efficient model: {most_efficient}")
        
        # Find model with most detailed responses
        if model_stats:
            most_detailed = max(model_stats.keys(), 
                              key=lambda m: model_stats[m]["avg_response_length"])
            recommendations.append(f"Most detailed responses: {most_detailed}")
        
            # Find most cost-effective (lowest token usage)
            least_tokens = min(model_stats.keys(),
                              key=lambda m: model_stats[m]["avg_input_tokens"] + model_stats[m]["avg_output_tokens"])
            recommendations.append(f"Most cost-effective: {least_tokens}")
        
        return recommendations
    
    def run_full_evaluation(self) -> str:
        """
        Run the complete promptfoo evaluation pipeline.
        
        Returns:
            Path to results directory
        """
        print("🎯 Starting PromptFoo Evaluation Pipeline...")
        
        # Load artifacts
        artifacts = self.load_inference_artifacts()
        if not artifacts:
            print("❌ No artifacts found to evaluate")
            return None
        
        # Extract test cases
        test_cases = self.extract_test_cases(artifacts)
        if not test_cases:
            print("❌ No valid test cases found in artifacts")
            return None
        
        print(f"📊 Extracted {len(test_cases)} test cases from {len(artifacts)} artifacts")
        
        # Create dataset and comparison
        dataset_file = self.create_promptfoo_dataset(test_cases)
        comparison_file = self.generate_comparison_report(test_cases)
        
        # Generate summary report
        summary = self.generate_summary_report(test_cases)
        
        # Save summary report
        summary_file = os.path.join(self.data_dir, "evaluation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Summary report saved: {summary_file}")
        
        # Print summary
        self._print_summary_report(summary)
        
        print(f"\n✅ PromptFoo evaluation pipeline completed!")
        print(f"📁 Data files available in: {self.data_dir}/")
        print(f"📁 Results will be saved to: {self.results_dir}/")
        
        return self.eval_dir
    
    def _print_summary_report(self, summary: Dict[str, Any]):
        """
        Print a formatted summary report.
        
        Args:
            summary: Summary report dictionary
        """
        print("\n" + "="*60)
        print("📊 PROMPTFOO EVALUATION SUMMARY")
        print("="*60)
        
        print(f"🕒 Timestamp: {summary.get('timestamp', 'Unknown')}")
        print(f"📝 Total Test Cases: {summary.get('total_test_cases', 0)}")
        print(f"🤖 Models Analyzed: {summary.get('models_analyzed', 0)}")
        
        model_stats = summary.get('model_statistics', {})
        if model_stats:
            print(f"\n📈 MODEL PERFORMANCE:")
            for model, stats in model_stats.items():
                print(f"  {model}:")
                print(f"    • Responses: {stats['response_count']}")
                print(f"    • Avg Input Tokens: {stats['avg_input_tokens']:.1f}")
                print(f"    • Avg Output Tokens: {stats['avg_output_tokens']:.1f}")
                print(f"    • Avg Response Length: {stats['avg_response_length']:.1f} chars")
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """
    Main function to run the promptfoo evaluation pipeline.
    """
    print("🚀 PromptFoo Evaluator for LLM Trainer POC")
    print("=" * 50)
    
    # Determine the correct artifacts directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(os.path.dirname(current_dir), "artifacts")
    
    evaluator = PromptFooEvaluator(artifacts_dir=artifacts_dir, eval_dir=current_dir)
    evaluator.run_full_evaluation()
    
    print("\n📋 Next steps:")
    print("1. Set your API keys: export OPENAI_API_KEY=... and GOOGLE_API_KEY=...")
    print("2. Run full evaluation with LLM as a judge: npx promptfoo eval -c promptfooconfig.yaml")
    print("3. View results: npx promptfoo view results/")


if __name__ == "__main__":
    main()
