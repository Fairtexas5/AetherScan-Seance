"""
ai_client.py
─────────────────────────────────────────────────────────
Ollama LLM client — tuned for MacBook Pro with Apple Silicon / Intel.

Recommended model: phi3.5:latest  (~2.7GB RAM, fast on Metal/CPU)

SETUP (one-time):
    1. Download Ollama: https://ollama.com/download
    2. Pull the model:  ollama pull phi3.5:latest
    3. Start server:    ollama serve
    4. Run pipeline:    python main.py

WHY phi3.5:latest:
    - Improved over phi3:mini — better instruction following & reasoning
    - Microsoft-optimized for efficient CPU/Metal inference
    - Larger context window (128K tokens) for complex prompts
    - Excellent quality for structured tasks like this

Expected LLM call times on MacBook Pro:
    - Validation check : ~5-15 seconds
    - Anomaly explain  : ~10-20 seconds
    - Final report     : ~20-40 seconds
─────────────────────────────────────────────────────────
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# ── Model config ─────────────────────────────────────────
MODEL_NAME   = "phi3.5:latest"       # Best for MacBook Pro
OLLAMA_URL   = "127.0.0.1:11434"
TEMPERATURE  = 0.1                   # Low = focused, factual responses
MAX_TOKENS   = 512                   # Shorter = faster inference
NUM_THREADS  = 6                     # Tune to your Mac's core count if needed
# ─────────────────────────────────────────────────────────


def get_llm() -> ChatOllama:
    """Return an Ollama LLM instance configured for MacBook Pro."""
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        num_thread=NUM_THREADS,      # Ollama param: number of CPU threads
    )


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Query local Ollama. Returns response text.
    Gracefully handles Ollama not running.
    """
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        print("    (waiting for local AI... ~10-30s on CPU)", flush=True)
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        msg = str(e)
        if "Connection refused" in msg or "connect" in msg.lower():
            return "[AI offline — run: ollama serve]"
        return f"[AI error: {msg}]"
