import asyncio
import atexit
import html
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import ffmpeg
import numpy as np
import soundfile as sf
import stripe
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
try:
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from posthog import Posthog
    from posthog.integrations.asgi import PosthogASGIMiddleware
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    # Billing and audio processing must remain available when optional analytics
    # packages are absent or deliberately omitted from a minimal deployment.
    OBSERVABILITY_AVAILABLE = False
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("duckit")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
LEGAL_DIR = BASE_DIR / "legal"
DEMO_DIR = BASE_DIR / "assets" / "demo"
# Tests and one-off tooling can point the engine at a disposable working
# directory so a local run never mutates the deployment database.
DATA_DIR = Path(os.environ.get("DUCKIT_DATA_DIR") or BASE_DIR / ".duckit-data")
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "private-outputs"
DB_PATH = DATA_DIR / "duckit.sqlite3"
MODEL_PATH = BASE_DIR / "models" / "silero_vad.jit"
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
FFPROBE_AVAILABLE = shutil.which("ffprobe") is not None

MAX_FREE_EXPORTS = 3
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
OUTPUT_TTL_SECONDS = 24 * 60 * 60
PCM_BLOCK_FRAMES = 65_536
VAD_SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512
COOKIE_NAME = "duckit_identity"
COOKIE_MAX_AGE = 365 * 24 * 60 * 60

# Accepted containers. MP4/MOV carry video too; only their first audio stream is
# decoded, so phone footage and editor exports can be used directly.
MEDIA_EXTENSIONS = {
    ".mp3": "MP3",
    ".mp2": "MP2",
    ".mpga": "MP3",
    ".wav": "WAV",
    ".wave": "WAV",
    ".m4a": "M4A",
    ".mp4": "MP4",
    ".m4v": "M4V",
    ".mov": "MOV",
    ".aac": "AAC",
    ".flac": "FLAC",
    ".ogg": "OGG",
    ".oga": "OGG",
    ".webm": "WebM",
}
MEDIA_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
    "audio/aac",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
    "video/mp4",
    "video/x-m4v",
    "video/quicktime",
    "video/webm",
    "application/ogg",
}

# The formats advertised to users; every one of them is exercised by the upload
# probe before any audio work starts.
SUPPORTED_FORMAT_LABELS = ["MP4", "MOV", "MP3", "WAV", "M4A"]

# Envelope shape: 0.3-0.5 s attacks and releases remove audible steps at every
# speech boundary without eating short utterances.
VAD_ENTER_THRESHOLD = 0.50
VAD_EXIT_THRESHOLD = 0.35
MIN_SPEECH_SECONDS = 0.06
# Energy window used to scale ducking depth, in dBFS.
RMS_FLOOR_DB = -45.0
RMS_CEIL_DB = -18.0
# Quiet-but-real speech still receives this share of the full ducking depth.
ADAPTIVE_FLOOR = 0.35
# One heavy job at a time keeps a small instance responsive; further jobs wait.
MAX_QUEUED_JOBS = 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_base_url: str = ""
    app_session_secret: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro_monthly: str = ""
    cookie_secure: bool | None = None
    posthog_project_token: str = ""
    posthog_host: str = ""
    environment: str = "development"
    # Public-facing copy and legal identity. Override per deployment so the
    # footer, policies and pricing never drift from the real merchant details.
    support_email: str = "support@duckit.app"
    company_name: str = "Duck It"
    pro_price_label: str = "£9/mo"
    legal_jurisdiction: str = "England and Wales"
    # Engine tuning. Length is bounded so one worker is never asked to analyse
    # an implausibly long file in a single request.
    max_source_seconds: int = 60 * 60
    envelope_attack_ms: int = 350
    envelope_release_ms: int = 450
    max_concurrent_jobs: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
APP_BASE_URL = settings.app_base_url.rstrip("/")
APP_SESSION_SECRET = settings.app_session_secret
STRIPE_SECRET_KEY = settings.stripe_secret_key
STRIPE_WEBHOOK_SECRET = settings.stripe_webhook_secret
STRIPE_PRICE_PRO_MONTHLY = settings.stripe_price_pro_monthly
# Clamped so a misconfigured environment cannot disable smoothing or ask a
# single worker to decode dozens of hours of audio.
MAX_SOURCE_SECONDS = min(4 * 60 * 60, max(60, settings.max_source_seconds))
ENVELOPE_ATTACK_MS = min(800, max(50, settings.envelope_attack_ms))
ENVELOPE_RELEASE_MS = min(800, max(50, settings.envelope_release_ms))
MAX_CONCURRENT_JOBS = min(4, max(1, settings.max_concurrent_jobs))
SUPPORT_EMAIL = settings.support_email.strip() or "support@duckit.app"
COMPANY_NAME = settings.company_name.strip() or "Duck It"
PRO_PRICE_LABEL = settings.pro_price_label.strip() or "£9/mo"
LEGAL_JURISDICTION = settings.legal_jurisdiction.strip() or "England and Wales"
# HTTPS deployments get secure cookies by default; localhost remains usable.
COOKIE_SECURE = settings.cookie_secure if settings.cookie_secure is not None else APP_BASE_URL.startswith("https://")
posthog_log_provider = None
posthog_log_logger = logging.getLogger("duckit.posthog_logs")
posthog_log_logger.setLevel(logging.INFO)
posthog_log_logger.propagate = False

if settings.posthog_project_token and settings.posthog_host and OBSERVABILITY_AVAILABLE:
    posthog_log_provider = LoggerProvider()
    posthog_log_exporter = OTLPLogExporter(
        endpoint=f"{settings.posthog_host.rstrip('/')}/i/v1/logs",
        headers={"Authorization": f"Bearer {settings.posthog_project_token}"},
    )
    posthog_log_provider.add_log_record_processor(
        BatchLogRecordProcessor(posthog_log_exporter)
    )
    posthog_log_logger.addHandler(LoggingHandler(logger_provider=posthog_log_provider))

posthog_client = None
if settings.posthog_project_token and settings.posthog_host and OBSERVABILITY_AVAILABLE:
    posthog_client = Posthog(
        settings.posthog_project_token,
        host=settings.posthog_host,
        enable_exception_autocapture=True,
    )
    atexit.register(posthog_client.shutdown)
elif settings.posthog_project_token and settings.posthog_host:
    logger.warning("PostHog is configured but its optional packages are not installed; analytics is disabled")
elif settings.environment != "production":
    if not settings.posthog_project_token:
        logger.warning(
            "POSTHOG_PROJECT_TOKEN variable required by PostHog is missing or un-configured, this causes events to be silently missed. This error stops appearing once POSTHOG_PROJECT_TOKEN is configured"
        )
    if not settings.posthog_host:
        logger.warning(
            "POSTHOG_HOST variable required by PostHog is missing or un-configured, this causes events to be silently missed. This error stops appearing once POSTHOG_HOST is configured"
        )

if not APP_BASE_URL or not APP_SESSION_SECRET:
    raise RuntimeError("APP_BASE_URL and APP_SESSION_SECRET must be set.")

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

stripe.api_key = STRIPE_SECRET_KEY
signer = URLSafeTimedSerializer(APP_SESSION_SECRET, salt="duckit-identity-v1")

# Silero VAD carries recurrent state between chunks. Serialising access prevents
# concurrent jobs from contaminating each other's speech decisions.
vad_lock = threading.Lock()
# Guard the (potentially slow, network-backed) model download so concurrent jobs
# cannot race each other through torch.hub.
vad_load_lock = threading.Lock()

# Heavy audio work is bounded: one job runs, a few wait, the rest are told to
# retry. Without this a burst of uploads starves the event loop on a small
# instance. The semaphore is created lazily so it always binds to the running
# event loop rather than the loop that happened to be active at import time.
processing_semaphore: "asyncio.Semaphore | None" = None
job_queue_depth = 0

# Models are intentionally kept process-global so every request reuses a single
# warm instance instead of reloading weights per job.
ml_models: dict[str, Any] = {}
vad_model: Any = None
vad_load_error: str | None = None


def _store_vad_model(model: Any, utils: Any, backend: str) -> None:
    global vad_model
    ml_models["model"] = model
    ml_models["utils"] = utils
    ml_models["backend"] = backend
    vad_model = model


def load_vad_model_safely() -> bool:
    """Load Silero VAD via torch.hub, preferring ONNX with a JIT fallback.

    Order of preference:
      1. ONNX runtime (fast + deterministic on CPU) via torch.hub.
      2. Standard TorchScript (JIT) model via torch.hub.
      3. A bundled local TorchScript file, for fully offline deployments.

    Every stage is isolated so a failure in one path never prevents the next.
    ``trust_repo=True`` suppresses torch.hub's interactive trust prompt, which
    would otherwise raise ``EOFError`` inside a non-interactive web worker.
    """
    global vad_model, vad_load_error
    if vad_model is not None:
        return True
    with vad_load_lock:
        # Another request may have loaded the model while we waited on the lock.
        if vad_model is not None:
            return True
        torch.set_num_threads(1)

        # Preferred path: ONNX runtime through torch.hub.
        try:
            logger.info("Loading Silero VAD (ONNX) via torch.hub ...")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=True,
                trust_repo=True,
                force_reload=False,
            )
            _store_vad_model(model, utils, "onnx")
            vad_load_error = None
            logger.info("Silero VAD ONNX model loaded successfully")
            return True
        except Exception as exc:
            logger.warning("ONNX Silero VAD load failed; falling back to JIT: %s", exc)

        # Fallback path: the standard TorchScript model through torch.hub.
        try:
            logger.info("Loading Silero VAD (JIT) via torch.hub ...")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=False,
                trust_repo=True,
                force_reload=False,
            )
            _store_vad_model(model, utils, "jit")
            vad_load_error = None
            logger.info("Silero VAD JIT model loaded successfully")
            return True
        except Exception as exc:
            logger.warning("torch.hub Silero VAD load failed; trying bundled model: %s", exc)

        # Last resort: the bundled TorchScript file (requires no network access).
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Local VAD model file not found at {MODEL_PATH}")
            logger.info("Loading bundled Silero VAD model from %s", MODEL_PATH)
            model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
            model.eval()
            _store_vad_model(model, None, "local")
            vad_load_error = None
            logger.info("Bundled Silero VAD model loaded successfully")
            return True
        except Exception as exc:
            vad_model = None
            ml_models.clear()
            vad_load_error = str(exc)
            logger.exception("Failed to load the Silero VAD model from any source")
            return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processing_semaphore, job_queue_depth
    # Rebind per loop: a semaphore created on a previous event loop would raise
    # when awaited, which matters for tests and --reload restarts.
    processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    job_queue_depth = 0
    load_vad_model_safely()
    if not FFMPEG_AVAILABLE:
        logger.error("FFmpeg executable was not found on PATH")
    if not FFPROBE_AVAILABLE:
        logger.error("FFprobe executable was not found on PATH")
    logger.info(
        "Engine ready: max %ss source, %d concurrent job(s), %.0f ms attack / %.0f ms release",
        MAX_SOURCE_SECONDS,
        MAX_CONCURRENT_JOBS,
        ENVELOPE_ATTACK_MS,
        ENVELOPE_RELEASE_MS,
    )
    if posthog_log_provider:
        posthog_log_logger.info(
            "audio_ducking_engine_started",
            extra={
                "event": "audio_ducking_engine_started",
                "max_source_seconds": MAX_SOURCE_SECONDS,
                "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                "ffmpeg_available": FFMPEG_AVAILABLE,
                "ffprobe_available": FFPROBE_AVAILABLE,
            },
        )
    yield
    if posthog_client:
        try:
            posthog_client.flush()
        except Exception:
            logger.exception("Could not flush PostHog events during shutdown")
    if posthog_log_provider:
        try:
            posthog_log_provider.force_flush(timeout_millis=5_000)
            posthog_log_provider.shutdown()
        except Exception:
            logger.exception("Could not flush PostHog logs during shutdown")
    global vad_model
    vad_model = None


app = FastAPI(
    title="DuckIt API",
    description="Automated AI Audio Ducking",
    lifespan=lifespan,
)

if posthog_client:
    app.add_middleware(PosthogASGIMiddleware, client=posthog_client)

allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", APP_BASE_URL).split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@contextmanager
def db():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db():
    with db() as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS identities (
                id TEXT PRIMARY KEY,
                stripe_customer_id TEXT UNIQUE,
                subscription_id TEXT UNIQUE,
                subscription_status TEXT NOT NULL DEFAULT 'none',
                free_exports_used INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL,
                output_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                is_free INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(identity_id) REFERENCES identities(id)
            );

            CREATE TABLE IF NOT EXISTS stripe_events (
                id TEXT PRIMARY KEY,
                processed_at INTEGER NOT NULL
            );

            -- Checkout Sessions are recorded at creation time so a delayed or
            -- missing webhook can still be reconciled on the next status poll.
            CREATE TABLE IF NOT EXISTS checkout_sessions (
                session_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
            );

            CREATE INDEX IF NOT EXISTS idx_checkout_identity
                ON checkout_sessions(identity_id, created_at DESC);
        """)
        cursor = conn.execute("PRAGMA table_info(identities)")
        columns = [col["name"] for col in cursor.fetchall()]
        if "free_exports_used" not in columns:
            conn.execute("ALTER TABLE identities ADD COLUMN free_exports_used INTEGER NOT NULL DEFAULT 0")


init_db()


def user_row(identity_id: str):
    with db() as conn:
        return conn.execute(
            "SELECT * FROM identities WHERE id = ?",
            (identity_id,),
        ).fetchone()


def is_pro(user) -> bool:
    return bool(user and user["subscription_status"] in {"active", "trialing"})


def set_identity_cookie(response: Response, identity_id: str):
    token = signer.dumps(identity_id)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def current_identity(request: Request, response: Response) -> str:
    identity_id = None
    token = request.cookies.get(COOKIE_NAME)

    if token:
        try:
            candidate = signer.loads(token, max_age=COOKIE_MAX_AGE)
            if user_row(candidate):
                identity_id = candidate
        except (BadSignature, SignatureExpired):
            pass

    if not identity_id:
        identity_id = uuid.uuid4().hex
        with db() as conn:
            conn.execute(
                "INSERT INTO identities (id, free_exports_used, created_at) VALUES (?, 0, ?)",
                (identity_id, int(time.time())),
            )
        set_identity_cookie(response, identity_id)

    if posthog_client:
        posthog_client.identify_context(identity_id)

    return identity_id


def entitlement_payload(identity_id: str):
    user = user_row(identity_id)
    pro = is_pro(user)
    used = user["free_exports_used"] if user else 0
    return {
        "plan": "pro" if pro else "free",
        "pro": pro,
        "free_exports_used": used,
        "max_free_exports": MAX_FREE_EXPORTS,
        "exports_remaining": max(0, MAX_FREE_EXPORTS - used),
        "subscription_status": user["subscription_status"] if user else "none",
    }


def feature_flags_for(identity_id: str) -> dict[str, Any]:
    """Evaluate every PostHog feature flag for an identity.

    Returns an empty mapping whenever PostHog is unavailable or the request
    fails, so callers can always fall back to their defaults without branching
    on configuration or network state.
    """
    if not posthog_client:
        return {}
    try:
        flags = posthog_client.get_all_flags(identity_id)
        return flags if isinstance(flags, dict) else {}
    except Exception:
        logger.exception("Could not evaluate PostHog feature flags for %s", identity_id)
        return {}


def experiment_value(identity_id: str, key: str, default: Any) -> Any:
    """Return a single feature-flag value, falling back to ``default``."""
    value = feature_flags_for(identity_id).get(key)
    return default if value is None else value


def require_billing_config(require_price: bool = False) -> None:
    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=503,
            detail={"code": "billing_unavailable", "message": "Stripe is not configured on this server."},
        )
    if not APP_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail={"code": "billing_unavailable", "message": "The public application URL is not configured."},
        )
    if require_price and not STRIPE_PRICE_PRO_MONTHLY.startswith("price_"):
        raise HTTPException(
            status_code=503,
            detail={"code": "billing_unavailable", "message": "The Pro monthly Stripe price is not configured."},
        )


def stripe_id(value) -> str | None:
    """Return an ID from a Stripe string or expanded object without assuming either shape."""
    if isinstance(value, str):
        return value
    if value is not None and hasattr(value, "get"):
        return value.get("id")
    return None


def stripe_error(operation: str, exc: Exception) -> HTTPException:
    logger.exception("Stripe failed while %s", operation)
    return HTTPException(
        status_code=502,
        detail={"code": "stripe_error", "message": f"Stripe could not complete {operation}. Please try again.", "operation": operation},
    )


# Status polling reconciles against Stripe at most once per identity per window:
# a late webhook must never turn every poll into a Stripe API call.
RECONCILE_INTERVAL_SECONDS = 4.0
reconcile_clock_lock = threading.Lock()
reconcile_attempts: dict[str, float] = {}


def should_reconcile(identity_id: str) -> bool:
    now = time.monotonic()
    with reconcile_clock_lock:
        last = reconcile_attempts.get(identity_id, 0.0)
        if now - last < RECONCILE_INTERVAL_SECONDS:
            return False
        reconcile_attempts[identity_id] = now
        if len(reconcile_attempts) > 10_000:
            # Entries only matter for a few seconds, so prune on growth.
            cutoff = now - RECONCILE_INTERVAL_SECONDS
            for key in [key for key, stamp in reconcile_attempts.items() if stamp < cutoff]:
                reconcile_attempts.pop(key, None)
        return True


def remember_checkout_session(session_id: str, identity_id: str) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO checkout_sessions
            (session_id, identity_id, created_at, status)
            VALUES (?, ?, ?, 'open')
            """,
            (session_id, identity_id, int(time.time())),
        )


def set_checkout_session_status(session_id: str, status: str) -> None:
    with db() as conn:
        conn.execute("UPDATE checkout_sessions SET status = ? WHERE session_id = ?", (status, session_id))


def latest_pending_checkout(identity_id: str, session_id: str | None = None) -> str | None:
    """Return a Checkout Session this browser may still be waiting on."""
    with db() as conn:
        if session_id:
            row = conn.execute(
                "SELECT session_id FROM checkout_sessions WHERE session_id = ? AND identity_id = ?",
                (session_id, identity_id),
            ).fetchone()
            if row:
                return row["session_id"]
        row = conn.execute(
            """
            SELECT session_id FROM checkout_sessions
            WHERE identity_id = ? AND status = 'open' AND created_at > ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (identity_id, int(time.time()) - 6 * 60 * 60),
        ).fetchone()
    return row["session_id"] if row else None


def reconcile_checkout_session(session_id: str, identity_id: str) -> bool:
    """Apply a completed Checkout Session; True when Pro is active afterwards."""
    try:
        session = stripe.checkout.Session.retrieve(session_id, expand=["subscription"]).to_dict()
    except Exception:
        logger.warning("Could not retrieve Checkout Session %s while reconciling", session_id, exc_info=True)
        return False

    user = user_row(identity_id)
    expected_customer = user["stripe_customer_id"] if user else None
    # The same ownership checks the /verify endpoint enforces: a session ID must
    # never grant Pro access to a different browser.
    if session.get("mode") != "subscription" or session.get("client_reference_id") != identity_id:
        return False
    if expected_customer and stripe_id(session.get("customer")) != expected_customer:
        return False

    state = session.get("status")
    if state != "complete":
        set_checkout_session_status(session_id, "expired" if state == "expired" else "open")
        return False

    subscription = session.get("subscription")
    if not subscription:
        return False
    if isinstance(subscription, str):
        try:
            subscription = stripe.Subscription.retrieve(subscription)
        except Exception:
            logger.warning("Could not retrieve subscription %s while reconciling", subscription, exc_info=True)
            return False
    try:
        save_subscription(subscription, identity_id)
    except Exception:
        logger.exception("Could not persist reconciled subscription for %s", identity_id)
        return False
    set_checkout_session_status(session_id, "complete")
    return is_pro(user_row(identity_id))


def reconcile_subscription(identity_id: str, subscription_id: str) -> bool:
    """Refresh a known subscription so renewals and cancellations self-heal."""
    try:
        subscription = stripe.Subscription.retrieve(subscription_id)
    except Exception:
        logger.warning("Could not retrieve subscription %s while reconciling", subscription_id, exc_info=True)
        return False
    try:
        save_subscription(subscription, identity_id)
    except Exception:
        logger.exception("Could not persist reconciled subscription for %s", identity_id)
        return False
    return is_pro(user_row(identity_id))


def stripe_event_seen(event_id: str) -> bool:
    with db() as conn:
        return conn.execute("SELECT 1 FROM stripe_events WHERE id = ?", (event_id,)).fetchone() is not None


def record_stripe_event(event_id: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO stripe_events (id, processed_at) VALUES (?, ?)",
            (event_id, int(time.time())),
        )


def reserve_free_export(identity_id: str, limit: int) -> bool:
    """Atomically claim one free export before the heavy work starts.

    A single conditional UPDATE keeps two concurrent uploads from both passing
    the entitlement check, and claiming up front means a failed job cannot burn
    quota silently.
    """
    with db() as conn:
        result = conn.execute(
            "UPDATE identities SET free_exports_used = free_exports_used + 1 "
            "WHERE id = ? AND free_exports_used < ?",
            (identity_id, limit),
        )
        return result.rowcount == 1


def release_export_reservation(identity_id: str) -> None:
    """Undo a free-export reservation after a failed job so usage is never lost."""
    with db() as conn:
        conn.execute(
            "UPDATE identities SET free_exports_used = MAX(free_exports_used - 1, 0) WHERE id = ?",
            (identity_id,),
        )


def cleanup_expired_outputs():
    now = int(time.time())
    with db() as conn:
        expired = conn.execute(
            "SELECT id, output_path FROM jobs WHERE expires_at < ?",
            (now,),
        ).fetchall()
        for job in expired:
            try:
                Path(job["output_path"]).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove expired output %s", job["id"])
        conn.execute("DELETE FROM jobs WHERE expires_at < ?", (now,))


def api_error(status_code: int, code: str, message: str, **extra: Any) -> HTTPException:
    """Build a structured API error so every client can render a stable shape."""
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update({key: value for key, value in extra.items() if value is not None})
    return HTTPException(status_code=status_code, detail=detail)


def format_duration_label(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    if total < 60:
        return f"{total} second{'s' if total != 1 else ''}"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, remainder = divmod(minutes, 60)
    return f"{hours} hr {remainder} min" if remainder else f"{hours} hr"


def validate_upload_metadata(upload: UploadFile) -> None:
    """Reject obviously unsupported files before a large body reaches disk."""
    name = (upload.filename or "").strip()
    suffix = Path(name).suffix.lower()
    if suffix in MEDIA_EXTENSIONS:
        return
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if not suffix and content_type in MEDIA_MIME_TYPES:
        return
    supported = ", ".join(sorted({label for label in MEDIA_EXTENSIONS.values()}))
    raise api_error(
        415,
        "unsupported_format",
        f"{name or 'That file'} is not a supported format. Use one of: {supported}.",
        filename=name or None,
    )


def _probe_duration(probe: dict[str, Any]) -> float:
    """Read duration from the container, falling back to individual streams.

    Fragmented MP4/MOV files frequently omit ``format.duration``, so stream
    durations must be considered rather than treated as a corrupt upload.
    """
    candidates = [probe.get("format", {}).get("duration")]
    for stream in probe.get("streams", []) or []:
        candidates.append(stream.get("duration"))
    best = 0.0
    for candidate in candidates:
        if candidate in (None, "N/A", ""):
            continue
        try:
            best = max(best, float(candidate))
        except (TypeError, ValueError):
            continue
    return best


def probe_media(
    path: Path,
    label: str,
    max_seconds: float | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Inspect an upload before processing and fail with a structured error."""
    if not FFPROBE_AVAILABLE:
        raise api_error(503, "ffprobe_unavailable", "Media inspection is unavailable on this server.")
    try:
        probe = ffmpeg.probe(str(path))
    except Exception as exc:
        logger.warning("Could not probe %s file %s: %s", label, path.name, exc)
        raise api_error(
            422,
            "unreadable_media",
            f"The {label} file could not be read. It may be corrupt or use an unsupported codec.",
            request_id=request_id,
        ) from exc

    streams = probe.get("streams", []) or []
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    duration = _probe_duration(probe)
    container = probe.get("format", {}).get("format_name")

    if not audio_streams:
        raise api_error(
            415,
            "no_audio_stream",
            f"The {label} file does not contain an audio track.",
            request_id=request_id,
            container=container,
        )
    if duration <= 0:
        raise api_error(
            422,
            "invalid_duration",
            f"Could not determine the length of the {label} file.",
            request_id=request_id,
            container=container,
        )
    if max_seconds is not None and duration > max_seconds + 0.5:
        raise api_error(
            413,
            "source_too_long",
            f"The {label} file is {format_duration_label(duration)} long; the limit is "
            f"{format_duration_label(max_seconds)}.",
            request_id=request_id,
            max_seconds=int(max_seconds),
            duration=round(duration, 2),
        )

    primary = audio_streams[0]
    return {
        "duration": duration,
        "container": container,
        "codec": primary.get("codec_name"),
        "channels": int(primary.get("channels") or 0),
        "sample_rate": int(primary.get("sample_rate") or 0),
        "has_video": any(stream.get("codec_type") == "video" for stream in streams),
        "audio_streams": len(audio_streams),
    }


def audio_channels(path: Path) -> int:
    """Best-effort channel count for an intermediate file (0 when unknown)."""
    try:
        probe = ffmpeg.probe(str(path))
    except Exception:
        return 0
    for stream in probe.get("streams", []) or []:
        if stream.get("codec_type") == "audio":
            try:
                return int(stream.get("channels") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


@asynccontextmanager
async def engine_slot():
    """Bound concurrent audio jobs so a burst of uploads cannot starve the loop."""
    global job_queue_depth, processing_semaphore
    if job_queue_depth >= MAX_CONCURRENT_JOBS + MAX_QUEUED_JOBS:
        raise api_error(
            429,
            "engine_busy",
            "The audio engine is busy with other jobs. Please retry in a few seconds.",
            retry_after_ms=15000,
        )
    job_queue_depth += 1
    try:
        if processing_semaphore is None:
            processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        async with processing_semaphore:
            yield
    finally:
        job_queue_depth -= 1


class AudioPipelineError(RuntimeError):
    """A safe, step-specific error for a failing external audio operation."""

    def __init__(self, step: str, message: str):
        super().__init__(message)
        self.step = step


def run_ffmpeg(stream, step: str) -> None:
    """Run FFmpeg and retain its diagnostic output in server logs, never the client."""
    try:
        stream.run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        logger.error("FFmpeg failed during %s: %s", step, stderr[-4_000:] or repr(exc))
        raise AudioPipelineError(step, "FFmpeg could not decode or encode the audio.") from exc
    except OSError as exc:
        logger.exception("Could not start FFmpeg during %s", step)
        raise AudioPipelineError(step, "FFmpeg is unavailable.") from exc


async def save_upload(upload: UploadFile, path: Path):
    """Persist an upload without blocking the event loop on disk writes."""
    total = 0
    output = await run_in_threadpool(path.open, "wb")
    try:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise api_error(
                    413,
                    "file_too_large",
                    "Each source file must be smaller than 250 MB.",
                    max_bytes=MAX_UPLOAD_BYTES,
                )
            await run_in_threadpool(output.write, chunk)
    finally:
        # Close before the request handler unlinks the file: Windows refuses to
        # delete a path that still has an open handle.
        await run_in_threadpool(output.close)
        await upload.close()


def prepare_vad_audio(input_path: Path, output_path: Path, duration: float | None = None):
    # Only the first audio stream is decoded: MP4/MOV uploads may carry large
    # video streams that must never enter the filter graph.
    output_kwargs = {
        "ar": 16000,
        "ac": 1,
        "format": "wav",
        "acodec": "pcm_s16le",
        "map": "0:a:0",
    }
    if duration is not None:
        output_kwargs["t"] = duration
    run_ffmpeg(
        ffmpeg.input(str(input_path)).output(str(output_path), **output_kwargs),
        "preparing 16 kHz VAD audio",
    )
    if not output_path.is_file() or output_path.stat().st_size < 44:
        raise AudioPipelineError("preparing 16 kHz VAD audio", "The converted dialogue track is empty.")


class SpeechRegion(NamedTuple):
    """A detected speech span in 16 kHz samples, with its relative energy."""

    start: int
    end: int
    strength: float


def _unpack_region(region) -> tuple[int, int, float]:
    """Accept both SpeechRegion and legacy (start, end) pairs."""
    if len(region) >= 3:
        return int(region[0]), int(region[1]), float(region[2])
    return int(region[0]), int(region[1]), 1.0


def frame_strength(frame: np.ndarray) -> float:
    """Map a frame's RMS level onto 0..1 so quiet speech ducks less deeply."""
    if frame.size == 0:
        return 0.0
    energy = np.asarray(frame, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(energy))))
    if rms <= 0.0:
        return 0.0
    decibels = 20.0 * float(np.log10(max(rms, 1e-6)))
    return float(np.clip((decibels - RMS_FLOOR_DB) / (RMS_CEIL_DB - RMS_FLOOR_DB), 0.0, 1.0))


def detect_speech_samples(vad_path: Path, threshold: float | None = None) -> list[SpeechRegion]:
    """Return speech regions in samples at the fixed 16 kHz VAD rate.

    Hysteresis (enter 0.50 / exit 0.35) stops the model chattering around a
    single threshold, and every region carries its mean frame energy so ducking
    depth can follow how loud the speaker actually is.
    """
    if vad_model is None:
        raise RuntimeError("VAD model is unavailable")
    samples, sample_rate = sf.read(str(vad_path), dtype="float32", always_2d=False)
    if sample_rate != VAD_SAMPLE_RATE:
        raise AudioPipelineError("reading VAD audio", f"Expected {VAD_SAMPLE_RATE} Hz dialogue audio, got {sample_rate} Hz")
    wav = torch.from_numpy(np.asarray(samples, dtype=np.float32).reshape(-1))
    if not len(wav):
        return []

    enter = VAD_ENTER_THRESHOLD if threshold is None else float(threshold)
    exit_level = min(enter, VAD_EXIT_THRESHOLD)
    min_samples = max(1, round(MIN_SPEECH_SECONDS * VAD_SAMPLE_RATE))

    regions: list[SpeechRegion] = []
    chunk_size = VAD_FRAME_SAMPLES  # Required frame size for the 16 kHz Silero JIT model.
    in_speech = False
    start_sample = 0
    strength_sum = 0.0
    strength_frames = 0

    def close_region(end_sample: int) -> None:
        end = max(start_sample, min(end_sample, len(wav)))
        if end - start_sample >= min_samples:
            mean_strength = strength_sum / strength_frames if strength_frames else 0.0
            regions.append(SpeechRegion(start_sample, end, mean_strength))

    # reset_states() and inference must be one critical section: this model is stateful.
    with vad_lock, torch.inference_mode():
        if hasattr(vad_model, "reset_states"):
            vad_model.reset_states()
        for frame_index, offset in enumerate(range(0, len(wav), chunk_size), start=1):
            chunk = wav[offset : offset + chunk_size]
            # Padding ensures a final partial frame is evaluated instead of dropped.
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
            probability = float(vad_model(chunk, 16000).item())
            if not in_speech and probability >= enter:
                in_speech = True
                start_sample = offset
                strength_sum = 0.0
                strength_frames = 0
            if in_speech:
                strength_sum += frame_strength(chunk.numpy())
                strength_frames += 1
                if probability < exit_level:
                    close_region(offset + VAD_FRAME_SAMPLES)
                    in_speech = False
            if frame_index % 2000 == 0:
                logger.info("VAD progress: %.1f%%", 100.0 * offset / max(1, len(wav)))
    if in_speech:
        close_region(len(wav))
    return regions


def merge_regions(regions: list, join_samples: int) -> list[SpeechRegion]:
    """Join neighbouring regions so fades never restore volume during speech.

    A joined region keeps a mean strength over its spoken samples only: bridging
    a short pause must not make real speech look quieter than it is.
    """
    merged: list[SpeechRegion] = []
    for region in regions:
        start, end, strength = _unpack_region(region)
        if end <= start:
            continue
        strength = float(np.clip(strength, 0.0, 1.0))
        if merged and start <= merged[-1].end + join_samples:
            previous = merged[-1]
            new_end = max(previous.end, end)
            previous_length = previous.end - previous.start
            added_length = max(1, new_end - max(previous.end, start))
            total_length = max(1, previous_length + added_length)
            blended = (previous.strength * previous_length + strength * added_length) / total_length
            merged[-1] = SpeechRegion(previous.start, new_end, float(np.clip(blended, 0.0, 1.0)))
        else:
            merged.append(SpeechRegion(start, end, strength))
    return merged


def music_regions(vad_regions: list, music_sr: int, join_seconds: float) -> list[SpeechRegion]:
    """Map VAD's fixed 16 kHz sample coordinates to the music stream exactly once."""
    scale = music_sr / VAD_SAMPLE_RATE
    join_samples = max(0, round(join_seconds * VAD_SAMPLE_RATE))
    mapped: list[SpeechRegion] = []
    for region in merge_regions(vad_regions, join_samples):
        start = round(region.start * scale)
        end = round(region.end * scale)
        if end > start:
            mapped.append(SpeechRegion(start, end, region.strength))
    return mapped


def smoothstep(progress):
    """Classic 3t² − 2t³ ease, used for every envelope ramp."""
    values = np.asarray(progress, dtype=np.float32)
    return values * values * (3.0 - 2.0 * values)


def region_target_gain(strength: float, duck_vol: float) -> float:
    """Ducking depth for one region: quiet speech ducks less than loud speech."""
    scaled = ADAPTIVE_FLOOR + (1.0 - ADAPTIVE_FLOOR) * float(np.clip(strength, 0.0, 1.0))
    return float(np.clip(1.0 - scaled * (1.0 - duck_vol), duck_vol, 1.0))


def gain_curve_for_block(
    block_start: int,
    frame_count: int,
    regions: list,
    attack_samples: int,
    release_samples: int,
    duck_vol: float,
) -> np.ndarray:
    """Return a bounded one-dimensional gain curve for exactly one PCM block.

    Each region ramps down over ``attack_samples``, holds its own adaptive
    target, then ramps back up over ``release_samples``. Ramps shrink for very
    short regions so the curve can never invert, and overlapping segments keep
    the quieter value so music is never amplified.
    """
    gain = np.ones(frame_count, dtype=np.float32)
    block_end = block_start + frame_count
    duck_vol = float(np.clip(duck_vol, 0.0, 1.0))
    attack_samples = max(1, int(attack_samples))
    release_samples = max(1, int(release_samples))
    for region in regions:
        speech_start, speech_end, strength = _unpack_region(region)
        if speech_end <= speech_start:
            continue
        # A ramp can never be longer than half of the region it belongs to.
        half_span = max(1, (speech_end - speech_start) // 2)
        attack = min(attack_samples, half_span)
        release = min(release_samples, half_span)
        target = region_target_gain(strength, duck_vol)
        segments = (
            (speech_start - attack, speech_start, 1.0, target),
            (speech_start, speech_end, target, target),
            (speech_end, speech_end + release, target, 1.0),
        )
        for segment_start, segment_end, first_gain, last_gain in segments:
            left, right = max(block_start, segment_start), min(block_end, segment_end)
            if right <= left:
                continue
            positions = np.arange(left, right, dtype=np.float32)
            span = max(1, segment_end - segment_start)
            progress = np.clip((positions - segment_start) / span, 0.0, 1.0)
            values = first_gain + (last_gain - first_gain) * smoothstep(progress)
            local_left, local_right = left - block_start, right - block_start
            gain[local_left:local_right] = np.minimum(gain[local_left:local_right], values)
    return np.clip(gain, duck_vol, 1.0)


def render_ducked_music(
    music_path: Path,
    vad_regions: list,
    output_path: Path,
    duck_vol: float,
    attack_seconds: float | None = None,
    release_seconds: float | None = None,
):
    """Stream music in bounded blocks: no whole-track matrices or shape mismatches."""
    attack_seconds = (ENVELOPE_ATTACK_MS / 1000.0) if attack_seconds is None else attack_seconds
    release_seconds = (ENVELOPE_RELEASE_MS / 1000.0) if release_seconds is None else release_seconds
    pcm_path = output_path.with_name(f"{output_path.stem}-source.wav")
    try:
        run_ffmpeg(
            # Only the first audio stream is decoded, so a video container's
            # picture track never reaches the sample buffer.
            ffmpeg.input(str(music_path)).output(
                str(pcm_path), acodec="pcm_s16le", format="wav", map="0:a:0"
            ),
            "decoding music for ducking",
        )
        with sf.SoundFile(str(pcm_path), mode="r") as source:
            if source.samplerate <= 0 or source.channels <= 0 or source.frames <= 0:
                raise AudioPipelineError("reading decoded music", "The music track contains no PCM audio.")
            attack_samples = max(1, round(attack_seconds * source.samplerate))
            release_samples = max(1, round(release_seconds * source.samplerate))
            # Pauses shorter than a release are bridged: restoring the music for
            # half a beat would be more distracting than leaving it down.
            regions = music_regions(vad_regions, source.samplerate, release_seconds)
            with sf.SoundFile(
                str(output_path), mode="w", samplerate=source.samplerate,
                channels=source.channels, format="WAV", subtype="PCM_16",
            ) as destination:
                frame_offset = 0
                while True:
                    block = source.read(PCM_BLOCK_FRAMES, dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    gain = gain_curve_for_block(
                        frame_offset, len(block), regions, attack_samples, release_samples, duck_vol
                    )
                    if gain.shape[0] != block.shape[0]:
                        raise AudioPipelineError("applying ducking curve", "Gain curve did not match decoded audio frames.")
                    ducked_block = block * gain[:, np.newaxis]
                    if not np.isfinite(ducked_block).all():
                        raise AudioPipelineError("applying ducking curve", "Decoded music contains non-finite samples.")
                    destination.write(ducked_block)
                    frame_offset += len(block)
    finally:
        pcm_path.unlink(missing_ok=True)


def mix_audio(dialogue_path: Path, ducked_music_path: Path, output_path: Path, max_seconds: float | None = None):
    """Run the final FFmpeg encode off the event loop."""
    # libmp3lame refuses streams whose channel layouts disagree, so the mix is
    # normalised explicitly: mono only when both sources are genuinely mono.
    music_channels = audio_channels(ducked_music_path)
    dialogue_channels = audio_channels(dialogue_path)
    output_channels = 1 if music_channels == 1 and dialogue_channels == 1 else 2
    mixed = ffmpeg.filter(
        [ffmpeg.input(str(ducked_music_path)), ffmpeg.input(str(dialogue_path))],
        "amix",
        inputs=2,
        duration="longest",
        dropout_transition=0.5,
        normalize=0,
    )
    mixed = ffmpeg.filter(
        mixed, "aformat", channel_layouts="mono" if output_channels == 1 else "stereo"
    )
    output_kwargs = {
        "acodec": "libmp3lame",
        "audio_bitrate": "320k",
        "ac": output_channels,
        "threads": 1,
    }
    if max_seconds is not None:
        output_kwargs["t"] = max_seconds
    run_ffmpeg(ffmpeg.output(mixed, str(output_path), **output_kwargs), "mixing and encoding output")


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-store"})


# /assets is deliberately not mounted: it holds working files from local runs.
# Only the two pre-rendered demo clips are published from this directory.
DEMO_ASSETS = {"original": "original.mp3", "ducked": "ducked.mp3"}


@app.get("/demo/{variant}", include_in_schema=False)
def demo_audio(variant: str):
    filename = DEMO_ASSETS.get(variant)
    if not filename:
        raise HTTPException(status_code=404, detail="Not found.")
    path = DEMO_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Demo audio is not available.")
    return FileResponse(
        path,
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


LEGAL_PAGES = {
    "terms": "terms.html",
    "privacy": "privacy.html",
    "refunds": "refunds.html",
}


def render_legal_page(filename: str) -> str:
    """Fill a legal template from deployment configuration, escaping values."""
    path = LEGAL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    document = path.read_text(encoding="utf-8")
    replacements = {
        "{{COMPANY_NAME}}": html.escape(COMPANY_NAME),
        "{{SUPPORT_EMAIL}}": html.escape(SUPPORT_EMAIL),
        "{{JURISDICTION}}": html.escape(LEGAL_JURISDICTION),
        "{{PRO_PRICE}}": html.escape(PRO_PRICE_LABEL),
        "{{YEAR}}": str(datetime.now(timezone.utc).year),
    }
    for token, value in replacements.items():
        document = document.replace(token, value)
    return document


def legal_route(page: str):
    filename = LEGAL_PAGES[page]
    return HTMLResponse(
        render_legal_page(filename),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/terms", include_in_schema=False)
def terms_page():
    return legal_route("terms")


@app.get("/privacy", include_in_schema=False)
def privacy_page():
    return legal_route("privacy")


@app.get("/refunds", include_in_schema=False)
def refunds_page():
    return legal_route("refunds")


@app.get("/api/health")
def health_check():
    if not FFMPEG_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "engine": "unavailable", "detail": "FFmpeg is not installed or not on PATH."},
        )
    if not FFPROBE_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "engine": "unavailable", "detail": "FFprobe is not installed or not on PATH."},
        )
    if vad_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "engine": "unavailable", "detail": vad_load_error},
        )
    return {
        "status": "online",
        "engine": "ready",
        "vad_backend": ml_models.get("backend"),
        "queue_depth": job_queue_depth,
    }


@app.get("/api/billing/status")
def billing_status(
    session_id: str | None = None,
    identity_id: str = Depends(current_identity),
):
    """Return entitlement, reconciling a pending Checkout the webhook missed.

    Webhook delivery can lag by seconds, and a misconfigured endpoint can drop
    events entirely. Rather than making the browser wait, the status poll is the
    self-healing path: it re-checks the newest pending Checkout Session (or the
    known subscription) against Stripe, throttled to one call per identity per
    window so polling stays cheap.
    """
    payload = entitlement_payload(identity_id)
    payload["pending_checkout"] = False
    payload["reconciling"] = False
    payload["retry_after_ms"] = 0
    if payload["pro"] or not STRIPE_SECRET_KEY:
        return payload

    if session_id and not session_id.startswith("cs_"):
        session_id = None
    candidate = latest_pending_checkout(identity_id, session_id)
    user = user_row(identity_id)
    subscription_id = user["subscription_id"] if user else None
    payload["pending_checkout"] = candidate is not None

    if (candidate or subscription_id) and should_reconcile(identity_id):
        payload["reconciling"] = True
        reconciled = (
            reconcile_checkout_session(candidate, identity_id)
            if candidate
            else reconcile_subscription(identity_id, subscription_id)
        )
        if reconciled:
            return entitlement_payload(identity_id)
        payload["pending_checkout"] = latest_pending_checkout(identity_id) is not None

    if payload["pending_checkout"] or payload["reconciling"]:
        payload["retry_after_ms"] = int(RECONCILE_INTERVAL_SECONDS * 1000)
    return payload


@app.get("/api/experiments")
def experiments(identity_id: str = Depends(current_identity)):
    """Return PostHog feature-flag variants for the connected identity.

    Evaluated server-side against the signed cookie identity so a single
    assignment can drive both UI copy and privileged logic (for example the
    free-export limit) for the same user.
    """
    return {"flags": feature_flags_for(identity_id), "identity_id": identity_id}


@app.get("/api/config")
def public_config():
    """Public bootstrap values for the frontend.

    The PostHog project token is a publishable key by design (it is embedded in
    client-side JavaScript), so it is safe to expose. No secrets (Stripe keys,
    session secret, webhook secret) are ever returned here.
    """
    return {
        "posthog_project_token": settings.posthog_project_token or "",
        "posthog_host": settings.posthog_host or "",
        "support_email": SUPPORT_EMAIL,
        "company_name": COMPANY_NAME,
        "pro_price_label": PRO_PRICE_LABEL,
        "max_source_seconds": MAX_SOURCE_SECONDS,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "supported_formats": SUPPORTED_FORMAT_LABELS,
        "envelope": {
            "attack_ms": ENVELOPE_ATTACK_MS,
            "release_ms": ENVELOPE_RELEASE_MS,
        },
        # The demo player hides itself when the clips have not been generated.
        "demo": (
            {"original": "/demo/original", "ducked": "/demo/ducked"}
            if (DEMO_DIR / "original.mp3").is_file() and (DEMO_DIR / "ducked.mp3").is_file()
            else None
        ),
    }


@app.post("/api/billing/verify")
def verify_checkout_session(session_id: str = Form(...), identity_id: str = Depends(current_identity)):
    """Synchronously reconcile a successful Checkout return before the webhook arrives."""
    require_billing_config()
    if not session_id.startswith("cs_"):
        raise HTTPException(status_code=422, detail={"code": "invalid_checkout_session", "message": "Invalid Checkout Session ID."})
    try:
        session = stripe.checkout.Session.retrieve(session_id, expand=["subscription"]).to_dict()
    except Exception as exc:
        raise stripe_error("Checkout verification", exc) from exc

    user = user_row(identity_id)
    expected_customer = user["stripe_customer_id"] if user else None
    if (
        session.get("mode") != "subscription"
        or session.get("client_reference_id") != identity_id
        or (expected_customer and stripe_id(session.get("customer")) != expected_customer)
    ):
        # A Checkout Session ID must never be usable to grant a different browser Pro access.
        raise HTTPException(status_code=403, detail="That Checkout session does not belong to this browser.")
    if session.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Checkout has not completed yet.")

    subscription = session.get("subscription")
    if not subscription:
        raise HTTPException(status_code=409, detail="Checkout did not create a subscription.")
    if isinstance(subscription, str):
        try:
            subscription = stripe.Subscription.retrieve(subscription)
        except Exception as exc:
            raise stripe_error("subscription activation", exc) from exc
    try:
        save_subscription(subscription, identity_id)
    except (sqlite3.Error, ValueError) as exc:
        logger.exception("Could not persist verified Stripe subscription")
        raise HTTPException(
            status_code=500,
            detail={"code": "billing_state_error", "message": "Payment was verified but subscription access could not be saved. Please contact support."},
        ) from exc
    set_checkout_session_status(session_id, "complete")
    if posthog_client:
        posthog_client.capture(
            "subscription_activated",
            properties={
                "plan": "pro",
                "activation_source": "checkout_verification",
            },
        )
    return entitlement_payload(identity_id)


@app.post("/api/billing/checkout")
def create_checkout(identity_id: str = Depends(current_identity)):
    require_billing_config(require_price=True)

    user = user_row(identity_id)
    if user is None:
        raise HTTPException(status_code=401, detail={"code": "identity_missing", "message": "Your browser session could not be loaded. Refresh and try again."})
    if is_pro(user):
        raise HTTPException(status_code=409, detail="Your Pro subscription is already active.")

    customer_id = user["stripe_customer_id"]
    if not customer_id:
        try:
            customer = stripe.Customer.create(metadata={"duckit_identity_id": identity_id})
        except Exception as exc:
            raise stripe_error("customer setup", exc) from exc
        customer_id = customer.id
        with db() as conn:
            conn.execute(
                "UPDATE identities SET stripe_customer_id = ? WHERE id = ?",
                (customer_id, identity_id),
            )

    try:
        checkout = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            client_reference_id=identity_id,
            line_items=[{"price": STRIPE_PRICE_PRO_MONTHLY, "quantity": 1}],
            allow_promotion_codes=True,
            success_url=f"{APP_BASE_URL}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{APP_BASE_URL}/?checkout=cancelled",
            metadata={"duckit_identity_id": identity_id},
            subscription_data={"metadata": {"duckit_identity_id": identity_id}},
        )
    except Exception as exc:
        raise stripe_error("Checkout creation", exc) from exc
    if not checkout.url:
        raise HTTPException(status_code=502, detail={"code": "stripe_error", "message": "Stripe did not return a Checkout URL."})
    # Recorded before the redirect so a dropped webhook can still be reconciled
    # the next time this browser asks for its billing status.
    remember_checkout_session(checkout.id, identity_id)
    if posthog_client:
        posthog_client.capture(
            "checkout_started",
            properties={"plan": "pro", "billing_interval": "monthly"},
        )
    return {"url": checkout.url}


@app.post("/api/billing/portal")
def create_billing_portal(identity_id: str = Depends(current_identity)):
    require_billing_config()
    user = user_row(identity_id)
    if not user or not user["stripe_customer_id"]:
        raise HTTPException(status_code=404, detail="No billing account exists yet.")

    try:
        portal = stripe.billing_portal.Session.create(
            customer=user["stripe_customer_id"],
            return_url=APP_BASE_URL,
        )
    except Exception as exc:
        raise stripe_error("billing portal creation", exc) from exc
    return {"url": portal.url}


@app.post("/api/duck", status_code=201)
async def duck_audio(
    dialogue: UploadFile = File(...),
    music: UploadFile = File(...),
    duck_vol: float = Form(0.15, ge=0.01, le=0.50),
    trim_seconds: int | None = Form(None),
    identity_id: str = Depends(current_identity),
):
    run_id = uuid.uuid4().hex
    dialogue_path = UPLOAD_DIR / f"{run_id}-dialogue.audio"
    music_path = UPLOAD_DIR / f"{run_id}-music.audio"
    vad_path = UPLOAD_DIR / f"{run_id}-vad.wav"
    ducked_music_path = UPLOAD_DIR / f"{run_id}-ducked-music.wav"
    output_path = OUTPUT_DIR / f"{run_id}.mp3"
    # Set only once quota is actually claimed, so cleanup can hand it back.
    reserved_free_export = False

    try:
        cleanup_expired_outputs()
        if not FFMPEG_AVAILABLE:
            raise api_error(503, "engine_unavailable", "FFmpeg is not installed or not on PATH.")
        if not FFPROBE_AVAILABLE:
            raise api_error(503, "engine_unavailable", "FFprobe is not installed or not on PATH.")
        # Reject unsupported containers from their metadata before writing bodies.
        validate_upload_metadata(dialogue)
        validate_upload_metadata(music)
        await save_upload(dialogue, dialogue_path)
        await save_upload(music, music_path)

        # Probe before extraction: only the first audio stream is ever decoded, so
        # a video-only file or an unreadable container fails here with a clear,
        # actionable message instead of halfway through the pipeline.
        dialogue_info = await run_in_threadpool(
            probe_media, dialogue_path, "dialogue", MAX_SOURCE_SECONDS, run_id
        )
        music_info = await run_in_threadpool(
            probe_media, music_path, "music", MAX_SOURCE_SECONDS, run_id
        )
        longest_source = max(dialogue_info["duration"], music_info["duration"])
        logger.info(
            "Audio job %s: dialogue %.1fs (%s/%s), music %.1fs (%s/%s)",
            run_id,
            dialogue_info["duration"],
            dialogue_info["container"],
            dialogue_info["codec"],
            music_info["duration"],
            music_info["container"],
            music_info["codec"],
        )

        user = user_row(identity_id)
        pro_active = is_pro(user)
        free_job = not pro_active

        free_export_limit = MAX_FREE_EXPORTS
        free_clip_seconds = 180.0
        if free_job:
            # Trial limits can be A/B tested via PostHog feature flags; the
            # constants above remain the authoritative defaults. Evaluate off the
            # event loop because the PostHog client performs blocking I/O.
            flags = await run_in_threadpool(feature_flags_for, identity_id)
            # Multivariate flags arrive as strings, so coerce defensively and
            # ignore booleans (a malformed flag must never become a limit of 1).
            raw_export_limit = flags.get("free_export_limit", MAX_FREE_EXPORTS)
            if isinstance(raw_export_limit, bool) or raw_export_limit is None:
                raw_export_limit = MAX_FREE_EXPORTS
            try:
                free_export_limit = max(0, int(raw_export_limit))
            except (TypeError, ValueError):
                free_export_limit = MAX_FREE_EXPORTS
            raw_clip_seconds = flags.get("free_clip_seconds", 180.0)
            if isinstance(raw_clip_seconds, bool) or raw_clip_seconds is None:
                raw_clip_seconds = 180.0
            try:
                free_clip_seconds = float(raw_clip_seconds)
            except (TypeError, ValueError):
                free_clip_seconds = 180.0
            if free_clip_seconds <= 0:
                # A non-positive cap would silently produce an empty export.
                free_clip_seconds = 180.0
            current_used = user["free_exports_used"] if user else 0
            if current_used >= free_export_limit:
                raise api_error(
                    402,
                    "upgrade_required",
                    f"You have used your {free_export_limit} free exports. Upgrade to Pro for unlimited exports.",
                    free_export_limit=free_export_limit,
                )

        # Claim quota atomically before the heavy work: two simultaneous uploads
        # can no longer both pass the check, and the claim is refunded on failure.
        if free_job:
            reserved_free_export = await run_in_threadpool(
                reserve_free_export, identity_id, free_export_limit
            )
            if not reserved_free_export:
                raise api_error(
                    402,
                    "upgrade_required",
                    f"You have used your {free_export_limit} free exports. Upgrade to Pro for unlimited exports.",
                    free_export_limit=free_export_limit,
                )

        # If startup initialisation was skipped or failed, attempt an emergency,
        # on-the-fly model load before giving up on this request.
        if vad_model is None:
            if not await run_in_threadpool(load_vad_model_safely):
                raise api_error(
                    503, "engine_unavailable", "The speech detector is unavailable. Please try again shortly."
                )

        # Free exports are always capped server-side; the browser's trim choice is
        # a convenience, never the enforcement mechanism.
        processing_duration = min(longest_source, free_clip_seconds) if free_job else None
        # Everything below is CPU bound and runs in worker threads, one job at a
        # time, so a burst of uploads cannot starve the event loop.
        async with engine_slot():
            await run_in_threadpool(prepare_vad_audio, dialogue_path, vad_path, processing_duration)
            try:
                speech_regions = await run_in_threadpool(detect_speech_samples, vad_path)
                analysed_seconds = processing_duration or dialogue_info["duration"]
                logger.info(
                    "Audio job %s: VAD found %d speech regions across %.1fs of dialogue",
                    run_id,
                    len(speech_regions),
                    analysed_seconds,
                )
                await run_in_threadpool(
                    render_ducked_music, music_path, speech_regions, ducked_music_path, duck_vol
                )
            except AudioPipelineError:
                raise
            except Exception as exc:
                logger.exception("Audio job %s failed while analysing or ducking", run_id)
                raise AudioPipelineError("applying ducking curve", "Audio analysis failed.") from exc

            await run_in_threadpool(
                mix_audio,
                dialogue_path,
                ducked_music_path,
                output_path,
                free_clip_seconds if free_job else None,
            )

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AudioPipelineError("encoding output", "The encoder did not create an audio file.")

        job_id = uuid.uuid4().hex
        filename = "duckit-mix.mp3"
        expires_at = int(time.time()) + OUTPUT_TTL_SECONDS

        # Quota was already claimed before processing, so only the job row is new.
        with db() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                (id, identity_id, output_path, filename, is_free, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, identity_id, str(output_path), filename, int(free_job), expires_at, int(time.time())),
            )

        updated_user = user_row(identity_id)
        exports_used = updated_user["free_exports_used"] if updated_user else 0

        if posthog_client:
            posthog_client.capture(
                "audio_ducking_completed",
                properties={
                    "plan": "free" if free_job else "pro",
                    "dialogue_duration_seconds": round(dialogue_info["duration"], 1),
                    "music_duration_seconds": round(music_info["duration"], 1),
                    "processed_duration_seconds": round(processing_duration or longest_source, 1),
                    "duck_volume": duck_vol,
                    "speech_region_count": len(speech_regions),
                    "free_exports_used": exports_used,
                },
            )
        if posthog_log_provider:
            posthog_log_logger.info(
                "audio_ducking_completed",
                extra={
                    "event": "audio_ducking_completed",
                    "plan": "free" if free_job else "pro",
                    "dialogue_duration_seconds": round(dialogue_info["duration"], 1),
                    "music_duration_seconds": round(music_info["duration"], 1),
                    "processed_duration_seconds": round(processing_duration or longest_source, 1),
                    "duck_volume": duck_vol,
                    "speech_region_count": len(speech_regions),
                    "free_exports_used": exports_used,
                },
            )

        return {
            "job_id": job_id,
            "filename": filename,
            "download_url": f"/api/jobs/{job_id}/download",
            "exports_remaining": max(0, free_export_limit - exports_used),
            "free_exports_used": exports_used,
            # Echoed so the UI can report the envelope it actually received.
            "envelope": {
                "attack_ms": ENVELOPE_ATTACK_MS,
                "release_ms": ENVELOPE_RELEASE_MS,
                "speech_regions": len(speech_regions),
            },
        }

    except HTTPException:
        output_path.unlink(missing_ok=True)
        if reserved_free_export:
            await run_in_threadpool(release_export_reservation, identity_id)
        raise
    except AudioPipelineError as exc:
        output_path.unlink(missing_ok=True)
        if reserved_free_export:
            await run_in_threadpool(release_export_reservation, identity_id)
        logger.exception("Audio job %s failed at %s", run_id, exc.step)
        raise HTTPException(
            status_code=422 if exc.step.startswith("reading") else 500,
            detail={
                "code": "audio_processing_failed",
                "message": "Could not process the supplied audio.",
                "step": exc.step,
                "request_id": run_id,
            },
        ) from exc
    except Exception:
        output_path.unlink(missing_ok=True)
        if reserved_free_export:
            await run_in_threadpool(release_export_reservation, identity_id)
        logger.exception("Audio job %s failed unexpectedly", run_id)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "audio_processing_failed",
                "message": "The audio engine failed unexpectedly.",
                "request_id": run_id,
            },
        )
    finally:
        for path in (dialogue_path, music_path, vad_path, ducked_music_path):
            path.unlink(missing_ok=True)


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str, response: Response, identity_id: str = Depends(current_identity)):
    with db() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ? AND identity_id = ?", (job_id, identity_id)).fetchone()

    if not job:
        raise HTTPException(status_code=404, detail="Output not found.")
    if job["expires_at"] < int(time.time()) or not Path(job["output_path"]).exists():
        raise HTTPException(status_code=410, detail="This output has expired.")
    if not job["is_free"] and not is_pro(user_row(identity_id)):
        raise HTTPException(status_code=402, detail={"code": "upgrade_required", "message": "Pro subscription required."})

    if posthog_client:
        posthog_client.capture(
            "audio_export_downloaded",
            properties={
                "plan": "free" if job["is_free"] else "pro",
                "output_format": "mp3",
            },
        )
    return FileResponse(job["output_path"], media_type="audio/mpeg", filename=job["filename"])


def save_subscription(subscription, identity_hint=None):
    metadata = subscription.get("metadata", {}) or {}
    identity_id = identity_hint or metadata.get("duckit_identity_id")
    customer_id = stripe_id(subscription.get("customer"))
    subscription_id = stripe_id(subscription)
    status = subscription.get("status", "none")
    if not subscription_id:
        raise ValueError("Stripe returned a subscription without an ID")

    with db() as conn:
        if identity_id:
            result = conn.execute(
                "UPDATE identities SET stripe_customer_id = COALESCE(?, stripe_customer_id), subscription_id = ?, subscription_status = ? WHERE id = ?",
                (customer_id, subscription_id, status, identity_id),
            )
            if result.rowcount != 1:
                raise ValueError("The Checkout browser identity no longer exists")
        else:
            conn.execute(
                "UPDATE identities SET subscription_id = ?, subscription_status = ? WHERE stripe_customer_id = ? OR subscription_id = ?",
                (subscription_id, status, customer_id, subscription_id),
            )
            identity = conn.execute(
                "SELECT id FROM identities WHERE stripe_customer_id = ? OR subscription_id = ?",
                (customer_id, subscription_id),
            ).fetchone()
            identity_id = identity["id"] if identity else None
    return identity_id


@app.post("/api/stripe/webhook", include_in_schema=False)
async def stripe_webhook(request: Request):
    if not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail={"code": "billing_unavailable", "message": "Stripe webhooks are not configured."})

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, signature, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        logger.warning("Rejected Stripe webhook with invalid signature: %s", exc)
        raise HTTPException(status_code=400, detail={"code": "invalid_webhook", "message": "Invalid Stripe webhook signature."}) from exc

    if await run_in_threadpool(stripe_event_seen, event["id"]):
        return {"received": True}

    event_type = event["type"]
    obj = event["data"]["object"]

    if event_type in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
        session_id = obj.get("id")
        subscription_id = stripe_id(obj.get("subscription"))
        if session_id:
            # A completed session may still need async payment confirmation; it
            # stays "open" in that case and the status poll keeps reconciling.
            status = "complete" if subscription_id else "open"
            await run_in_threadpool(set_checkout_session_status, session_id, status)
        if subscription_id:
            try:
                subscription = await run_in_threadpool(stripe.Subscription.retrieve, subscription_id)
                await run_in_threadpool(save_subscription, subscription, obj.get("client_reference_id"))
            except Exception:
                # Return a 5xx: Stripe will safely retry because the event has not been marked seen.
                logger.exception("Could not apply Stripe checkout event %s", event["id"])
                raise HTTPException(status_code=500, detail={"code": "webhook_processing_failed", "message": "Could not apply the Stripe subscription event."})
    elif event_type in {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}:
        try:
            await run_in_threadpool(save_subscription, obj)
        except Exception:
            logger.exception("Could not apply Stripe subscription event %s", event["id"])
            raise HTTPException(status_code=500, detail={"code": "webhook_processing_failed", "message": "Could not apply the Stripe subscription event."})
    elif event_type in {"invoice.paid", "invoice.payment_failed", "invoice.payment_action_required"}:
        # Renewals and dunning never touch entitlement without this: refresh the
        # subscription so a lapse or recovery is reflected immediately.
        subscription_id = stripe_id(obj.get("subscription"))
        if subscription_id:
            try:
                subscription = await run_in_threadpool(stripe.Subscription.retrieve, subscription_id)
                await run_in_threadpool(save_subscription, subscription)
            except Exception:
                logger.exception("Could not apply Stripe invoice event %s", event["id"])
                raise HTTPException(status_code=500, detail={"code": "webhook_processing_failed", "message": "Could not apply the Stripe subscription event."})
    elif event_type == "checkout.session.expired":
        session_id = obj.get("id")
        if session_id:
            await run_in_threadpool(set_checkout_session_status, session_id, "expired")

    await run_in_threadpool(record_stripe_event, event["id"])

    return {"received": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)

