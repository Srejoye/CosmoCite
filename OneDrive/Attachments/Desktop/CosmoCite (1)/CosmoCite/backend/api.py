"""FastAPI backend. Run (from project root): uvicorn api:app --app-dir backend --reload --port 8000"""
import functools, json, logging, os, threading, time
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from typing import Literal
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import llm
from data import CFG, ROOT, PROC

REP = ROOT / "reports"
REP.mkdir(exist_ok=True)
DIST = ROOT / "frontend" / "dist"
CSP = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self' data:"
CACHE_SIZE = 256
log = logging.getLogger("cosmocite")
answers = OrderedDict()   # finished answers, oldest first
answers_lock = threading.Lock()
ask_slots = threading.BoundedSemaphore(CFG["api"]["max_concurrent"])  # answers are CPU-heavy, so people wait briefly in line
ready = False
waiting = 0  # requests queued for an answer slot


@asynccontextmanager
async def lifespan(_):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    threading.Thread(target=warm, daemon=True).start()  # the server accepts requests while models load
    yield


def allowed_origins():
    """Deployments set COSMOCITE_ORIGINS. The site and API share an origin, so this only matters for other sites."""
    env = os.environ.get("COSMOCITE_ORIGINS")
    return [o.strip() for o in env.split(",") if o.strip()] if env else CFG["api"]["origins"]


app = FastAPI(title="CosmoCite", lifespan=lifespan)
router = APIRouter(prefix="/api")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(), allow_methods=["*"], allow_headers=["*"])
app.mount("/api/reports", StaticFiles(directory=REP), name="reports")


@app.middleware("http")
async def headers_and_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    log.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


def load(name):
    p = PROC / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() and p.stat().st_size else []


@functools.lru_cache(maxsize=1)
def _retriever():
    from retrieval import Retriever
    return Retriever()


def retriever():
    try:
        return _retriever()
    except FileNotFoundError as e:
        raise HTTPException(503, f"Search index not built, run backend/features.py first: {e}")


def cached(key):
    with answers_lock:
        if key in answers:
            answers.move_to_end(key)
            return answers[key]
    return None


@contextmanager
def answer_slot():
    """Heavy work runs one at a time. A short line forms, capped so waiters cannot exhaust the server's threads."""
    global waiting
    busy = HTTPException(429, "The server is busy answering other questions.")
    with answers_lock:
        if waiting >= CFG["api"]["max_waiting"]:
            raise busy
        waiting += 1
    try:
        got = ask_slots.acquire(timeout=CFG["api"]["queue_seconds"])
    finally:
        with answers_lock:
            waiting -= 1
    if not got:
        raise busy
    try:
        yield
    finally:
        ask_slots.release()


def warm():
    """Loads every model up front so the first visitor does not pay for it."""
    global ready
    try:
        from features import embed
        from retrieval import _cross_encoder
        _retriever()
        embed(["warm"])
        _cross_encoder()
    except (FileNotFoundError, OSError, ImportError):
        return  # /ask reports the problem when it is used
    ready = True
    try:
        llm.complete("Hi", max_tokens=1)
    except llm.LLMUnavailable:
        pass  # answers fall back to flan-T5


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    decoding: Literal["greedy", "beam", "top_k", "nucleus"] = "greedy"
    mode: Literal["bm25", "dense", "hybrid", "rerank"] | None = None
    engine: Literal["rank", "synth", "flan"] = "rank"

    @field_validator("question")
    @classmethod
    def has_text(cls, v):
        if not v.strip():
            raise ValueError("question is blank")
        return v.strip()


class Analyze(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@functools.lru_cache(maxsize=1)
def _passages():
    p = PROC / "chunks.jsonl"
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


@router.get("/stats")
def stats():
    return {"papers": len(load("papers.json")), "passages": _passages()}


@router.get("/papers")
def papers(organism: str = "", stressor: str = "", tissue: str = "", year_min: int = 0, year_max: int = 9999, q: str = ""):
    out = []
    for p in load("papers.json"):
        e = p["entities"]
        if organism and organism not in e["organism"]: continue
        if stressor and stressor not in e["stressor"]: continue
        if tissue and tissue not in e["tissue"]: continue
        if not (year_min <= (p["year"] or 0) <= year_max): continue
        if q and q.lower() not in p["title"].lower(): continue
        out.append(p)
    return out


@router.get("/papers/{paper_id}/related")
def related(paper_id: str, k: int = Query(5, ge=1, le=20)):
    try:
        return retriever().related(paper_id, k)
    except KeyError:
        raise HTTPException(404, "Paper not found")


@router.get("/gaps")
def gaps():
    return load("gaps.json")


@router.get("/health")
def health():
    return {"ready": ready}


@router.post("/ask")
def ask(r: Ask):
    from generate import answer
    repeatable = r.engine != "flan" or r.decoding in ("greedy", "beam")  # sampling should differ on every run
    key = (" ".join(r.question.lower().split()), r.decoding, r.mode, r.engine)
    if repeatable and (hit := cached(key)):
        return hit
    with answer_slot():
        if repeatable and (hit := cached(key)):  # someone ahead in the line may have just asked the same thing
            return hit
        try:
            out = answer(r.question, r.decoding, retriever(), r.mode, r.engine)
        except OSError as e:  # the image ships only what the default engine needs
            raise HTTPException(503, "This engine is not available on this server.") from e
        if repeatable and out["engine"] == r.engine:  # a fallback must not outlive the outage
            with answers_lock:
                answers[key] = out
                while len(answers) > CACHE_SIZE:
                    answers.popitem(last=False)
    return out


@router.get("/search")
def search(q: str = Query(min_length=1, max_length=500), k: int = Query(5, ge=1, le=50)):
    from retrieval import MODES
    with answer_slot():
        r = retriever()
        return {m: r.search(q, m, k) for m in MODES}


@router.post("/analyze")
def analyze(r: Analyze):
    from classify import predict
    from ner import predict_ner
    try:
        section = predict(r.text)
    except FileNotFoundError:  # entities still work: predict_ner falls back to the rule tagger
        section = {"label": None, "model": None, "probs": {}, "attention": [], "attention_model": None, "attention_label": None}
    return {**section, "entities": predict_ner(r.text)}


@router.get("/results")
def results():
    out = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in REP.glob("*.json")}
    out["images"] = sorted(p.name for p in REP.glob("*.png"))
    return out


app.include_router(router)


@app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def site(path: str = ""):
    """Serves the built frontend, so one process runs the whole app."""
    if path.startswith("api/") or "\x00" in path:
        raise HTTPException(404, "Not found")
    root = DIST.resolve()
    page, file = root / "index.html", (root / path).resolve()
    if path and file != page and file.is_relative_to(root) and file.is_file():
        return FileResponse(file)
    if not page.is_file():
        raise HTTPException(404, "Frontend not built. Run npm run build in frontend/.")
    return FileResponse(page, headers={"Content-Security-Policy": CSP})
