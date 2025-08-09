# PromptFoo Evaluation for LLM Trainer POC

This directory contains the complete PromptFoo evaluation setup for analyzing and comparing LLM inference results with general response quality metrics.

## Quick Start

### Option 1: Interactive Script
```bash
./run.sh
```

### Option 2: Manual Commands
```bash
# Setup (from project root)
uv run eval/setup.py

# Extract inference results from artifacts
uv run eval/extract_responses.py

# Run analysis without API calls
uv run eval/promptfoo_evaluator.py  

# Run full evaluation with API calls
cd eval
npx promptfoo eval -c promptfooconfig.yaml

# View results
npx promptfoo view results/
```

## Directory Structure

```
eval/
├── package.json                  # Node.js dependencies
├── package-lock.json            # Lock file for dependencies  
├── node_modules/                # PromptFoo dependencies
├── promptfooconfig.yaml         # PromptFoo configuration
├── promptfoo_evaluator.py       # Main evaluation script
├── setup.py                     # Setup and validation script
├── run.sh                       # Interactive runner script
├── README.md                    # This file
├── data/                        # Generated analysis files
│   ├── setup_analysis.json
│   ├── dataset.json
│   └── comparison_results.json
└── results/                     # Evaluation results
    └── eval_results.json
```

## Configuration

The evaluation is configured to test **general response quality** only:

- **Response Quality**: Length, completeness, clarity, and overall grade
- **General Reliability**: Detects error responses, refusals, and uncertainty
- **Cost Efficiency**: Tracks token usage and API costs
- **Model Comparison**: Side-by-side evaluation of different models
- **Format Compliance**: Ensures responses meet expected format requirements

**No domain-specific content**: All test cases are generic and don't contain any specialized domain information.

## Environment Variables

Set these before running evaluations:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## Scripts

### `setup.py`
- Validates existing inference artifacts
- Confirms general PromptFoo configuration exists  
- Analyzes artifact data and generates setup reports

### `promptfoo_evaluator.py`
- Main evaluation pipeline
- Processes artifacts directly from `../artifacts/`
- Generates comparison reports and analysis
- Works completely offline (no API calls required)
- Creates test datasets from your existing inference results

### `promptfooconfig.yaml`
- **Generic PromptFoo configuration** for general quality evaluation
- **No domain-specific content**: Uses only generic prompts and test cases
- Used when you want to test general response capabilities with live API calls
- Contains general quality assertions only
- Requires API keys for live testing

### `run.sh`
- Interactive menu for all operations
- Checks dependencies and API keys
- Provides guided evaluation workflow

## Evaluation Criteria

### Quality Assertions
- **Minimum Length**: Responses must be substantive (30+ characters)
- **No Empty Responses**: Ensures all models respond appropriately
- **Error Detection**: Identifies failed responses, refusals, and uncertainty
- **Response Grading**: Uses PromptFoo's grading system for quality assessment
- **Cost Control**: Keeps API costs within reasonable limits
- **Format Compliance**: Ensures responses meet length and format requirements

### Model Comparison
- **Token Efficiency**: Input/output token ratios
- **Response Quality**: Length, completeness, and grade scores
- **Cost Effectiveness**: Cost per quality unit
- **Reliability**: Error rates and consistency
- **General Performance**: Overall response quality without domain bias

## Integration with Main Pipeline

This evaluation system integrates seamlessly with your existing LLM training pipeline:

1. **Input**: Reads from `../artifacts/inference_results_*.json`
2. **Processing**: Uses your `uv` Python environment
3. **Output**: Generates results in `results/` directory
4. **Comparison**: Analyzes existing model responses vs new evaluations

## Troubleshooting

### Common Issues

**No artifacts found**: Run `uv run ../src/inference.py` first to generate inference results.

**PromptFoo not found**: Install with `npm install promptfoo` from project root.

**API errors**: Check that environment variables are set correctly.

**Permission denied**: Make run script executable with `chmod +x run.sh`.

### Debug Mode

Run with verbose output:
```bash
uv run --verbose eval/promptfoo_evaluator.py
npx promptfoo eval -c promptfooconfig.yaml --no-progress-bar
```

## Advanced Usage

### Custom Configurations
Edit `promptfooconfig.yaml` to:
- Add new models or providers
- Modify evaluation criteria  
- Change test cases or assertions
- Adjust cost thresholds

### Batch Evaluation
Process multiple artifact sets:
```bash
# Evaluate all artifacts at once
find ../artifacts -name "*.json" | xargs -I {} npx promptfoo eval -c promptfooconfig.yaml --model-outputs {}
```

### Export Results
Generate reports in different formats:
```bash
npx promptfoo eval -c promptfooconfig.yaml -o results/report.html
npx promptfoo eval -c promptfooconfig.yaml -o results/report.csv
```

## Next Steps

1. **Run Setup**: `uv run eval/setup.py`
2. **Set API Keys**: Export your OpenAI and Google API keys
3. **Run Evaluation**: `npx promptfoo eval -c promptfooconfig.yaml`
4. **View Results**: `npx promptfoo view results/`
5. **Iterate**: Modify configuration and re-run as needed
