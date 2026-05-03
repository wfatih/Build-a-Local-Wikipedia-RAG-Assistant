$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "Pulling Ollama models (requires 'ollama serve' running in another terminal)..."
ollama pull llama3.2:3b
ollama pull nomic-embed-text
Write-Host "Optional comparison model:"
try { ollama pull phi3:mini } catch { Write-Host "phi3:mini pull skipped." }

Write-Host ""
Write-Host "Setup complete. Next:"
Write-Host "  python -m src.ingest.run_ingest"
Write-Host "  streamlit run src/ui/streamlit_app.py    # or:  python -m src.ui.cli"
