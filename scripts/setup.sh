#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Pulling Ollama models (requires 'ollama serve' running in another terminal)…"
ollama pull llama3.2:3b
ollama pull nomic-embed-text
echo "Optional comparison model:"
ollama pull phi3:mini || true

echo
echo "Setup complete. Next:"
echo "  python -m src.ingest.run_ingest"
echo "  streamlit run src/ui/streamlit_app.py    # or:  python -m src.ui.cli"
