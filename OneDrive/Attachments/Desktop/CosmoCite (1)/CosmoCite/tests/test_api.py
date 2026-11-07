import threading
import pytest
from fastapi.testclient import TestClient
import api


@pytest.fixture
def client():
    return TestClient(api.app)


def test_unknown_decoding_is_a_client_error(client):
    assert client.post("/api/ask", json={"question": "q", "decoding": "psychic"}).status_code == 422


def test_empty_question_is_a_client_error(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_search_rejects_an_oversized_k(client):
    assert client.get("/api/search", params={"q": "bone", "k": 5000}).status_code == 422


def test_missing_index_reports_how_to_build_it(client, monkeypatch):
    def missing():
        raise FileNotFoundError("chunk_emb.npy")
    monkeypatch.setattr(api, "_retriever", missing)
    r = client.get("/api/search", params={"q": "bone"})
    assert r.status_code == 503 and "features.py" in r.json()["detail"]


def test_papers_filter_by_organism(client):
    rows = client.get("/api/papers", params={"organism": "mouse"}).json()
    assert rows and all("mouse" in p["entities"]["organism"] for p in rows)


def test_results_lists_reports(client):
    body = client.get("/api/results").json()
    assert "retrieval" in body and isinstance(body["images"], list)


def test_unknown_engine_is_a_client_error(client):
    assert client.post("/api/ask", json={"question": "q", "engine": "psychic"}).status_code == 422


def test_stats_reports_live_corpus_size(client):
    body = client.get("/api/stats").json()
    assert body["papers"] == len(api.load("papers.json")) and body["passages"] > body["papers"]


def fake_answer(calls, engine="rank"):
    def run(question, decoding, retriever, mode, asked):
        calls.append(question)
        return {"answer": "a", "sufficient": True, "citations": [], "decoding": decoding, "engine": engine, "reason": None, "evidence": None}
    return run


@pytest.fixture
def stubbed(client, monkeypatch):
    import generate
    calls = []
    api.answers.clear()
    monkeypatch.setattr(api, "_retriever", lambda: object())
    monkeypatch.setattr(generate, "answer", fake_answer(calls))
    return client, calls


def test_repeat_questions_are_served_from_the_cache(stubbed):
    client, calls = stubbed
    client.post("/api/ask", json={"question": "Bone loss?"})
    client.post("/api/ask", json={"question": "  bone   LOSS? "})
    assert len(calls) == 1


def test_sampled_decoding_is_never_cached(stubbed):
    client, calls = stubbed
    for _ in range(2):
        client.post("/api/ask", json={"question": "q", "engine": "flan", "decoding": "nucleus"})
    assert len(calls) == 2


def test_fallback_answers_are_not_cached(client, monkeypatch):
    import generate
    calls = []
    api.answers.clear()
    monkeypatch.setattr(api, "_retriever", lambda: object())
    monkeypatch.setattr(generate, "answer", fake_answer(calls, engine="flan"))
    for _ in range(2):
        client.post("/api/ask", json={"question": "q"})
    assert len(calls) == 2


def test_cache_drops_the_oldest_entry_when_full(stubbed, monkeypatch):
    client, calls = stubbed
    monkeypatch.setattr(api, "CACHE_SIZE", 1)
    for q in ("one", "two", "one"):
        client.post("/api/ask", json={"question": q})
    assert len(calls) == 3


def test_blank_question_is_a_client_error(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422


def test_warm_survives_a_missing_index(monkeypatch):
    def missing():
        raise FileNotFoundError("chunk_emb.npy")
    monkeypatch.setattr(api, "_retriever", missing)
    api.warm()


def test_health_reports_warm_state(client, monkeypatch):
    monkeypatch.setattr(api, "ready", True)
    assert client.get("/api/health").json() == {"ready": True}


def test_ask_turns_people_away_when_the_server_is_full(stubbed, monkeypatch):
    client, calls = stubbed
    monkeypatch.setitem(api.CFG["api"], "queue_seconds", 0)
    monkeypatch.setattr(api, "ask_slots", threading.BoundedSemaphore(1))
    api.ask_slots.acquire()
    assert client.post("/api/ask", json={"question": "q"}).status_code == 429 and calls == []


def test_a_full_server_still_serves_cached_answers(stubbed, monkeypatch):
    client, calls = stubbed
    client.post("/api/ask", json={"question": "q"})
    monkeypatch.setattr(api, "ask_slots", threading.BoundedSemaphore(1))
    api.ask_slots.acquire()
    assert client.post("/api/ask", json={"question": "q"}).status_code == 200


def test_a_failed_answer_frees_its_slot(client, monkeypatch):
    import generate

    def boom(*a):
        raise RuntimeError("boom")

    api.answers.clear()
    monkeypatch.setattr(api, "_retriever", lambda: object())
    monkeypatch.setattr(generate, "answer", boom)
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "ask_slots", slots)
    with pytest.raises(RuntimeError):
        client.post("/api/ask", json={"question": "q"})
    assert slots.acquire(blocking=False)


def test_only_known_sites_may_call_the_api(client):
    assert "access-control-allow-origin" not in client.get("/api/stats", headers={"Origin": "https://evil.example"}).headers
    assert client.get("/api/stats", headers={"Origin": "http://localhost:5173"}).headers["access-control-allow-origin"] == "http://localhost:5173"


def test_rank_is_an_accepted_engine(stubbed):
    client, calls = stubbed
    assert client.post("/api/ask", json={"question": "q", "engine": "rank"}).status_code == 200


def test_ranked_quotes_are_the_default_engine():
    assert api.Ask(question="q").engine == "rank"



@pytest.fixture
def site(client, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>home</html>")
    (dist / "assets" / "app.js").write_text("js")
    (tmp_path / "secret.txt").write_text("secret")
    monkeypatch.setattr(api, "DIST", dist)
    return client


def test_site_serves_built_files(site):
    assert site.get("/assets/app.js").text == "js"


def test_site_serves_the_app_for_page_routes(site):
    for path in ("/", "/ask", "/search", "/results"):
        assert site.get(path).text == "<html>home</html>"


def test_site_cannot_read_outside_its_folder(site):
    assert "secret" not in site.get("/..%2Fsecret.txt").text


def test_unknown_api_paths_stay_errors(site):
    assert site.get("/api/nope").status_code == 404


def test_missing_build_is_a_clear_error(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DIST", tmp_path / "missing")
    assert client.get("/").status_code == 404


def test_responses_carry_security_headers(client):
    h = client.get("/api/stats").headers
    assert h["x-content-type-options"] == "nosniff" and h["referrer-policy"] == "same-origin"


def test_origins_can_be_set_for_deployment(monkeypatch):
    monkeypatch.setenv("COSMOCITE_ORIGINS", "https://a.example, https://b.example")
    assert api.allowed_origins() == ["https://a.example", "https://b.example"]


def test_site_policy_allows_the_embedded_fonts(site):
    assert "font-src 'self' data:" in site.get("/").headers["content-security-policy"]


def test_ask_waits_for_a_free_slot(stubbed, monkeypatch):
    client, calls = stubbed
    monkeypatch.setitem(api.CFG["api"], "queue_seconds", 5)
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "ask_slots", slots)
    slots.acquire()
    threading.Timer(0.1, slots.release).start()
    assert client.post("/api/ask", json={"question": "q"}).status_code == 200



def test_a_waiting_request_uses_an_answer_finished_while_it_queued(stubbed, monkeypatch):
    client, calls = stubbed
    monkeypatch.setitem(api.CFG["api"], "queue_seconds", 5)
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "ask_slots", slots)
    slots.acquire()

    def finish():
        api.answers[("q", "greedy", None, "rank")] = {"answer": "done", "engine": "rank"}
        slots.release()

    threading.Timer(0.1, finish).start()
    assert client.post("/api/ask", json={"question": "q"}).json()["answer"] == "done" and calls == []


def test_too_many_waiters_are_turned_away_at_once(stubbed, monkeypatch):
    client, calls = stubbed
    monkeypatch.setattr(api, "waiting", api.CFG["api"]["max_waiting"])
    assert client.post("/api/ask", json={"question": "q"}).status_code == 429 and calls == []


def test_search_shares_the_queue_with_answers(client, monkeypatch):
    monkeypatch.setitem(api.CFG["api"], "queue_seconds", 0)
    monkeypatch.setattr(api, "_retriever", lambda: object())
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "ask_slots", slots)
    slots.acquire()
    assert client.get("/api/search", params={"q": "bone"}).status_code == 429


def test_a_missing_engine_model_is_a_clear_error(client, monkeypatch):
    import generate

    def missing(*a):
        raise OSError("flan-t5-base not found")

    api.answers.clear()
    monkeypatch.setattr(api, "_retriever", lambda: object())
    monkeypatch.setattr(generate, "answer", missing)
    assert client.post("/api/ask", json={"question": "q", "engine": "flan"}).status_code == 503


def test_site_answers_head_requests(site):
    assert site.head("/").status_code == 200


def test_every_route_to_the_page_carries_the_policy(site):
    assert "content-security-policy" in site.get("/index.html").headers


def test_a_null_byte_path_is_not_a_crash(site):
    assert site.get("/%00").status_code == 404


class FakeRelated:
    def related(self, paper_id, k):
        if paper_id == "nope":
            raise KeyError(paper_id)
        return [{"paper_id": "0002", "title": "T", "pmc_id": ""}][:k]


def test_related_papers_endpoint(client, monkeypatch):
    monkeypatch.setattr(api, "_retriever", FakeRelated)
    assert client.get("/api/papers/0001/related").json()[0]["paper_id"] == "0002"


def test_related_papers_of_an_unknown_paper_is_not_found(client, monkeypatch):
    monkeypatch.setattr(api, "_retriever", FakeRelated)
    assert client.get("/api/papers/nope/related").status_code == 404

