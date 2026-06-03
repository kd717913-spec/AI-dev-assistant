"""
QyverixAI — Backend API
FastAPI application with middleware, rate limiting, and full analysis engine.
"""

import os
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .routers import (
    analyze,
    auth,
    chat,
    debugging,
    explanation,
    history,
    share,
    subscribe,
    suggestions,
    upload_file,
    user_data,
)
from .services import database
from .services.scheduler import start_scheduler, stop_scheduler
from .database import Base, engine
from .schemas import HealthResponse


# ── Rate limiter config ───────────────────────────────────────────────────────
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", 30))
WINDOW_SECONDS = 60

rate_limit_store: dict[str, list[float]] = defaultdict(list)


# ── Rate limiter middleware ───────────────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        now = time.time()
        window_start = now - WINDOW_SECONDS

        # Remove timestamps outside current window
        rate_limit_store[ip] = [
            t for t in rate_limit_store[ip] if t > window_start
        ]

        request_count = len(rate_limit_store[ip])
        reset_time = int(now) + WINDOW_SECONDS  # unix timestamp of window end

        if request_count >= RATE_LIMIT:
            return Response(
                content='{"detail": "Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit":     str(RATE_LIMIT),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset":     str(reset_time),
                },
            )

        # Record this request before processing
        rate_limit_store[ip].append(now)
        remaining = max(0, RATE_LIMIT - len(rate_limit_store[ip]))

        response = await call_next(request)

        # Inject headers on every successful response
        response.headers["X-RateLimit-Limit"]     = str(RATE_LIMIT)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"]     = str(reset_time)

        return response


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()
    logging.getLogger(__name__).info("🚀 QyverixAI backend starting…")
    Base.metadata.create_all(bind=engine)
    start_scheduler()
    yield
    stop_scheduler()
    logging.getLogger(__name__).info("🛑 QyverixAI backend shutting down…")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QyverixAI",
    description="AI-powered developer assistant — code explanation, debugging, and improvement.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware (order matters — added last runs first) ────────────────────────
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Cache header middleware ───────────────────────────────────────────────────
@app.middleware("http")
async def add_cache_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/analyze/" and request.method == "POST":
        response.headers.setdefault("X-Cache", "MISS")
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(explanation.router,  prefix="/explanation",  tags=["Explanation"])
app.include_router(debugging.router,    prefix="/debugging",    tags=["Debugging"])
app.include_router(suggestions.router,  prefix="/suggestions",  tags=["Suggestions"])
app.include_router(analyze.router,      prefix="/analyze",      tags=["Full Analysis"])
app.include_router(subscribe.router,    prefix="/subscribe",    tags=["Subscription"])
app.include_router(history.router,      prefix="/history",      tags=["History"])
app.include_router(upload_file.router,  prefix="/upload",       tags=["Upload File"])
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(share.router)
app.include_router(user_data.router)


# ── Core endpoints ────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["System"])
async def root():
    return {
        "status": "ok",
        "version": "3.0.0",
        "message": "QyverixAI API is running.",
        "endpoints": [
            "/explanation/", "/debugging/", "/suggestions/", "/analyze/",
            "/subscribe/", "/share/", "/history/", "/upload/",
            "/auth/signup", "/auth/login", "/auth/me",
            "/chat/", "/user/",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "version": "3.0.0",
        "message": "QyverixAI is healthy.",
        "endpoints": [
            "/explanation/", "/debugging/", "/suggestions/", "/analyze/",
            "/subscribe/", "/share/", "/history/", "/upload/",
            "/auth/signup", "/auth/login", "/auth/me",
            "/chat/", "/user/",
        ],
    }


@app.get("/ping", tags=["System"])
async def ping():
    return {"message": "pong"}


# ── Static / Frontend ─────────────────────────────────────────────────────────
_frontend = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend):
    app.mount("/app", StaticFiles(directory=_frontend, html=True), name="frontend")


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
