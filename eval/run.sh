#!/bin/bash
# PromptFoo Evaluation Runner for LLM Trainer POC

echo "🚀 PromptFoo Evaluation Runner"
echo "=============================="

# Change to eval directory
cd "$(dirname "$0")"

# Check if promptfoo is installed
if ! npx promptfoo --version &> /dev/null; then
    echo "❌ PromptFoo not found. Installing..."
    npm install promptfoo
fi

echo ""
echo "📋 Available Commands:"
echo "1. Setup evaluation environment"
echo "2. Run artifact analysis (no API calls)"
echo "3. Run full PromptFoo evaluation (requires API keys)"
echo "4. View evaluation results"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "🔧 Setting up evaluation environment..."
        cd ..
        uv run eval/setup.py
        ;;
    2)
        echo "📊 Running artifact analysis..."
        cd ..
        uv run eval/promptfoo_evaluator.py
        ;;
    3)
        echo "🚀 Running full PromptFoo evaluation..."
        if [[ -z "$OPENAI_API_KEY" ]] || [[ -z "$GOOGLE_API_KEY" ]]; then
            echo "⚠️  Warning: API keys not found in environment"
            echo "Set OPENAI_API_KEY and GOOGLE_API_KEY before running evaluation"
            echo ""
            echo "Example:"
            echo "export OPENAI_API_KEY='your-key-here'"
            echo "export GOOGLE_API_KEY='your-key-here'"
            echo ""
            read -p "Continue anyway? (y/N): " confirm
            if [[ $confirm != [yY] ]]; then
                exit 0
            fi
        fi
        npx promptfoo eval -c promptfooconfig.yaml
        ;;
    4)
        echo "🌐 Opening results viewer..."
        npx promptfoo view results/
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac
