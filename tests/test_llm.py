import io
import json
import urllib.error
import pytest
import llm


def test_complete_sends_deterministic_options_and_returns_text(monkeypatch):
    sent = {}

    def fake(req, timeout):
        sent.update(json.loads(req.data))
        sent["timeout"] = timeout
        return io.BytesIO(b'{"response": "  Bone fell. [1]  "}')

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    assert llm.complete("p") == "Bone fell. [1]"
    assert sent["stream"] is False and sent["options"]["temperature"] == 0
    assert sent["keep_alive"] == llm.CFG["llm"]["keep_alive"]
    assert sent["model"] == llm.CFG["llm"]["model"] and sent["timeout"] == llm.CFG["llm"]["timeout"]


def test_network_failure_becomes_llm_unavailable(monkeypatch):
    def fake(req, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    with pytest.raises(llm.LLMUnavailable):
        llm.complete("p")


def test_timeout_becomes_llm_unavailable(monkeypatch):
    def fake(req, timeout):
        raise TimeoutError()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    with pytest.raises(llm.LLMUnavailable):
        llm.complete("p")


def test_malformed_reply_becomes_llm_unavailable(monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b"not json"))
    with pytest.raises(llm.LLMUnavailable):
        llm.complete("p")
