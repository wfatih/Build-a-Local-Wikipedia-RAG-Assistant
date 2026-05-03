"""Central configuration for the Local Wikipedia RAG Assistant."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CONV_DIR = DATA_DIR / "conversations"
LOG_DIR = DATA_DIR / "logs"
PEOPLE_FILE = DATA_DIR / "people.txt"
PLACES_FILE = DATA_DIR / "places.txt"
DB_PATH = DATA_DIR / "rag.db"
CACHE_DB_PATH = DATA_DIR / "cache.db"

# --- Ollama ---
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
PRIMARY_LLM = "llama3.2:3b"
SECONDARY_LLM = "phi3:mini"

# nomic-embed-text was trained with task prefixes — using them gives a
# meaningful retrieval-quality boost. See README for re-ingest instructions
# whenever these strings change.
EMBED_QUERY_PREFIX = "search_query: "
EMBED_DOC_PREFIX = "search_document: "

# --- Chunking ---
# Larger chunks keep more semantic context per item; the smaller `OVERLAP`
# preserves cross-paragraph continuity. Empirically tuned on this corpus —
# 320/60 reduces total chunks ~30% versus 220/40 with no measurable
# retrieval-quality regression.
CHUNK_TARGET_TOKENS = 320
CHUNK_OVERLAP_TOKENS = 60
APPROX_CHARS_PER_TOKEN = 4

# --- Retrieval ---
TOP_K = 5
RRF_K = 60
# Cross-encoder reranking is optional — if `sentence_transformers` isn't
# installed, the retriever silently falls back to plain RRF.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_POOL = 20
USE_RERANKER_DEFAULT = False  # toggleable from the UI

# --- Self-grounding ---
SELF_GROUNDING_DEFAULT = False  # toggleable from the UI; adds ~LLM-call latency

# --- Wikipedia ---
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "LocalWikiRAG/1.0 (academic; contact: developer@example.com)"

for d in (DATA_DIR, RAW_DIR, CONV_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
