"""Client for a local Ollama server. Free and offline once the model is pulled."""
import json, os, urllib.error, urllib.request
from data import CFG


class LLMUnavailable(Exception):
    """Ollama is not running, timed out, or returned something unreadable."""


def complete(prompt, max_tokens=None):
    c = CFG["llm"]
    host = os.environ.get("OLLAMA_HOST_URL", c["host"])
    body = {"model": c["model"], "prompt": prompt, "stream": False, "keep_alive": c["keep_alive"],
            "options": {"temperature": 0, "num_predict": max_tokens or c["max_tokens"], "num_ctx": c["num_ctx"]}}
    req = urllib.request.Request(f"{host}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=c["timeout"]))["response"].strip()
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
        raise LLMUnavailable(str(e)) from e
