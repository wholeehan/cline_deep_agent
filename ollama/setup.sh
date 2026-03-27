#!/bin/bash
# Build the custom qwen3-coder-tools model for tool-calling support.
#
# This wraps qwen3-coder:latest with a Qwen3 ChatML template that includes
# tool-calling markers ({{ .Tools }}, {{ .ToolCalls }}), enabling Ollama's
# native tool support without changing the model weights.
#
# Usage:
#   bash ollama/setup.sh
#
# After building, set in .env:
#   OLLAMA_SUBAGENT_MODEL=qwen3-coder-tools:latest

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="qwen3-coder-tools:latest"
BASE_MODEL="qwen3-coder:latest"

echo "=== Building custom Ollama model: $MODEL_NAME ==="
echo "Base model: $BASE_MODEL"
echo ""

# Verify base model exists
if ! ollama list 2>/dev/null | grep -q "$BASE_MODEL"; then
    echo "Base model $BASE_MODEL not found. Pulling..."
    ollama pull "$BASE_MODEL"
fi

# Build custom model
echo "Creating $MODEL_NAME from Modelfile..."
ollama create "$MODEL_NAME" -f "$SCRIPT_DIR/Modelfile.qwen3-coder-tools"

echo ""
echo "=== Model $MODEL_NAME created successfully ==="
echo ""
echo "To use it, set in your .env file:"
echo "  OLLAMA_SUBAGENT_MODEL=$MODEL_NAME"
echo ""
echo "Verify tool support:"
echo "  curl -s http://localhost:11434/api/chat -d '{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"test\",\"description\":\"test\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}}}}}],\"stream\":false}'"
