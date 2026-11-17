import atexit
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf
import stripe
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
DATA_DIR = BASE_DIR / ".duckit-data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "private-outputs"
DB_PATH = DATA_DIR / "duckit.sqlite3"
MODEL_PATH = BASE_DIR / "models" / "silero_vad.jit"
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

MAX_FREE_EXPORTS = 3
MAX_SOURCE_SECONDS = 60 * 60
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
OUTPUT_TTL_SECONDS = 24 * 60 * 60
PCM_BLOCK_FRAMES = 65_536
VAD_SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512
COOKIE_NAME = "duckit_identity"
COOKIE_MAX_AGE = 365 * 24 * 60 * 60

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
APP_BASE_URL = settings.app_base_url.rstrip("/")
APP_SESSION_SECRET = settings.app_session_secret
STRIPE_SECRET_KEY = settings.stripe_secret_key
STRIPE_WEBHOOK_SECRET = settings.stripe_webhook_secret
STRIPE_PRICE_PRO_MONTHLY = settings.stripe_price_pro_monthly
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

if not APP_BASE_URL or not APP_SESSION_SECRET:
    raise RuntimeError("APP_BASE_URL and APP_SESSION_SECRET must be set.")

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

stripe.api_key = STRIPE_SECRET_KEY
signer = URLSafeTimedSerializer(APP_SESSION_SECRET, salt="duckit-identity-v1")

vad_model: torch.jit.ScriptModule | None = None
vad_load_error: str | None = None
# Silero VAD carries recurrent state between chunks. Serialising access prevents
# concurrent jobs from contaminating each other's speech decisions.
vad_lock = threading.Lock()


def load_vad_model_safely() -> bool:
    """Load only the bundled TorchScript model; never use torch.hub/GitHub."""
    global vad_model, vad_load_error
    torch.set_num_threads(1)
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Local VAD model file not found at {MODEL_PATH}")
        logger.info("Loading local Silero VAD model from %s", MODEL_PATH)
        model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
        model.eval()
        vad_model = model
        vad_load_error = None
        logger.info("Local Silero VAD model loaded successfully")
        return True
    except Exception as exc:
        vad_model = None
        vad_load_error = str(exc)
        logger.exception("Failed to load the local Silero VAD model")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vad_model_safely()
    if not FFMPEG_AVAILABLE:
        logger.error("FFmpeg executable was not found on PATH")
    yield
    if posthog_client:
        try:
            posthog_client.flush()
        except Exception:
            logger.exception("Could not flush PostHog events during shutdown")
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


def source_duration(path: Path) -> float:
    try:
        probe = ffmpeg.probe(str(path))
        duration = float(probe["format"].get("duration", 0))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="One of the uploaded files is not a readable audio file.",
        ) from exc

    if duration <= 0:
        raise HTTPException(
            status_code=422,
            detail="Could not determine audio duration.",
        )
    return duration


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
    total = 0
    try:
        with path.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Each source file must be smaller than 250 MB.",
                    )
                output.write(chunk)
    finally:
        await upload.close()


def prepare_vad_audio(input_path: Path, output_path: Path, duration: float | None = None):
    output_kwargs = {"ar": 16000, "ac": 1, "format": "wav", "acodec": "pcm_s16le"}
    if duration is not None:
        output_kwargs["t"] = duration
    run_ffmpeg(
        ffmpeg.input(str(input_path)).output(str(output_path), **output_kwargs),
        "preparing 16 kHz VAD audio",
    )
    if not output_path.is_file() or output_path.stat().st_size < 44:
        raise AudioPipelineError("preparing 16 kHz VAD audio", "The converted dialogue track is empty.")


def detect_speech_samples(vad_path: Path, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Return speech regions in samples at the fixed 16 kHz VAD rate."""
    if vad_model is None:
        raise RuntimeError("VAD model is unavailable")
    samples, sample_rate = sf.read(str(vad_path), dtype="float32", always_2d=False)
    if sample_rate != VAD_SAMPLE_RATE:
        raise AudioPipelineError("reading VAD audio", f"Expected {VAD_SAMPLE_RATE} Hz dialogue audio, got {sample_rate} Hz")
    wav = torch.from_numpy(np.asarray(samples, dtype=np.float32).reshape(-1))
    if not len(wav):
        return []

    regions: list[tuple[int, int]] = []
    chunk_size = VAD_FRAME_SAMPLES  # Required frame size for the 16 kHz Silero JIT model.
    in_speech = False
    start_sample = 0
    # reset_states() and inference must be one critical section: this model is stateful.
    with vad_lock, torch.inference_mode():
        if hasattr(vad_model, "reset_states"):
            vad_model.reset_states()
        for offset in range(0, len(wav), chunk_size):
            chunk = wav[offset : offset + chunk_size]
            # Padding ensures a final partial frame is evaluated instead of dropped.
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
            probability = float(vad_model(chunk, 16000).item())
            if probability >= threshold and not in_speech:
                in_speech = True
                start_sample = offset
            elif probability < threshold and in_speech:
                regions.append((start_sample, min(offset, len(wav))))
                in_speech = False
    if in_speech:
        regions.append((start_sample, len(wav)))
    return regions


def merge_regions(regions: list[tuple[int, int]], join_samples: int) -> list[tuple[int, int]]:
    """Join neighbouring VAD frames so fades never restore volume during speech."""
    merged: list[list[int]] = []
    for start, end in regions:
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + join_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def music_regions(vad_regions: list[tuple[int, int]], music_sr: int, fade_duration: float) -> list[tuple[int, int]]:
    """Map VAD's fixed 16 kHz sample coordinates to the music stream exactly once."""
    scale = music_sr / VAD_SAMPLE_RATE
    return [
        (round(start * scale), round(end * scale))
        for start, end in merge_regions(vad_regions, round(fade_duration * VAD_SAMPLE_RATE))
        if end > start
    ]


def gain_curve_for_block(
    block_start: int,
    frame_count: int,
    regions: list[tuple[int, int]],
    fade_samples: int,
    duck_vol: float,
) -> np.ndarray:
    """Return a bounded one-dimensional gain curve for exactly one PCM block."""
    gain = np.ones(frame_count, dtype=np.float32)
    block_end = block_start + frame_count
    for speech_start, speech_end in regions:
        # Fade down before speech, hold while speech is present, then fade up.
        segments = (
            (speech_start - fade_samples, speech_start, 1.0, duck_vol),
            (speech_start, speech_end, duck_vol, duck_vol),
            (speech_end, speech_end + fade_samples, duck_vol, 1.0),
        )
        for segment_start, segment_end, first_gain, last_gain in segments:
            left, right = max(block_start, segment_start), min(block_end, segment_end)
            if right <= left:
                continue
            positions = np.arange(left, right, dtype=np.float32)
            span = max(1, segment_end - segment_start)
            values = first_gain + (last_gain - first_gain) * ((positions - segment_start) / span)
            local_left, local_right = left - block_start, right - block_start
            # In a fade overlap, the quieter curve wins; never amplify source audio.
            gain[local_left:local_right] = np.minimum(gain[local_left:local_right], values)
    return np.clip(gain, duck_vol, 1.0)


def render_ducked_music(music_path: Path, vad_regions: list[tuple[int, int]], output_path: Path, duck_vol: float):
    """Stream music in bounded blocks: no whole-track matrices or shape mismatches."""
    pcm_path = output_path.with_name(f"{output_path.stem}-source.wav")
    try:
        run_ffmpeg(
            ffmpeg.input(str(music_path)).output(str(pcm_path), acodec="pcm_s16le", format="wav"),
            "decoding music for ducking",
        )
        with sf.SoundFile(str(pcm_path), mode="r") as source:
            if source.samplerate <= 0 or source.channels <= 0 or source.frames <= 0:
                raise AudioPipelineError("reading decoded music", "The music track contains no PCM audio.")
            fade_samples = max(1, round(0.05 * source.samplerate))
            regions = music_regions(vad_regions, source.samplerate, 0.05)
            with sf.SoundFile(
                str(output_path), mode="w", samplerate=source.samplerate,
                channels=source.channels, format="WAV", subtype="PCM_16",
            ) as destination:
                frame_offset = 0
                while True:
                    block = source.read(PCM_BLOCK_FRAMES, dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    gain = gain_curve_for_block(frame_offset, len(block), regions, fade_samples, duck_vol)
                    if gain.shape[0] != block.shape[0]:
                        raise AudioPipelineError("applying ducking curve", "Gain curve did not match decoded audio frames.")
                    ducked_block = block * gain[:, np.newaxis]
                    if not np.isfinite(ducked_block).all():
                        raise AudioPipelineError("applying ducking curve", "Decoded music contains non-finite samples.")
                    destination.write(ducked_block)
                    frame_offset += len(block)
    finally:
        pcm_path.unlink(missing_ok=True)


def mix_audio(dialogue_path: Path, ducked_music_path: Path, output_path: Path, free_job: bool):
    """Run the final FFmpeg encode off the event loop."""
    mixed = ffmpeg.filter(
        [ffmpeg.input(str(ducked_music_path)), ffmpeg.input(str(dialogue_path))],
        "amix",
        inputs=2,
        duration="longest",
        dropout_transition=0.5,
        normalize=0,
    )
    output_kwargs = {"acodec": "libmp3lame", "audio_bitrate": "320k"}
    if free_job:
        output_kwargs["t"] = 180
    run_ffmpeg(ffmpeg.output(mixed, str(output_path), **output_kwargs), "mixing and encoding output")


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health_check():
    if not FFMPEG_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "engine": "unavailable", "detail": "FFmpeg is not installed or not on PATH."},
        )
    if vad_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "engine": "unavailable", "detail": vad_load_error},
        )
    return {"status": "online", "engine": "ready"}


@app.get("/api/billing/status")
def billing_status(identity_id: str = Depends(current_identity)):
    return entitlement_payload(identity_id)


@app.post("/api/billing/verify")
def verify_checkout_session(session_id: str = Form(...), identity_id: str = Depends(current_identity)):
    """Synchronously reconcile a successful Checkout return before the webhook arrives."""
    require_billing_config()
    if not session_id.startswith("cs_"):
        raise HTTPException(status_code=422, detail={"code": "invalid_checkout_session", "message": "Invalid Checkout Session ID."})
    try:
        session = stripe.checkout.Session.retrieve(session_id, expand=["subscription"])
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

    try:
        cleanup_expired_outputs()
        if not FFMPEG_AVAILABLE:
            raise HTTPException(status_code=503, detail="FFmpeg is not installed or not on PATH.")
        await save_upload(dialogue, dialogue_path)
        await save_upload(music, music_path)

        dialogue_seconds = await run_in_threadpool(source_duration, dialogue_path)
        music_seconds = await run_in_threadpool(source_duration, music_path)
        longest_source = max(dialogue_seconds, music_seconds)
        user = user_row(identity_id)
        pro_active = is_pro(user)
        free_job = not pro_active

        if longest_source > MAX_SOURCE_SECONDS:
            raise HTTPException(status_code=413, detail="Source tracks must be one hour or shorter.")

        if free_job:
            current_used = user["free_exports_used"] if user else 0
            if current_used >= MAX_FREE_EXPORTS:
                raise HTTPException(
                    status_code=402,
                    detail={
                        "code": "upgrade_required",
                        "message": "You have used your 3 free exports. Upgrade to Pro for unlimited exports.",
                    },
                )

        if vad_model is None:
            raise HTTPException(status_code=503, detail="The VAD engine is unavailable. Please try again shortly.")

        # Free exports are always capped server-side; the browser's trim choice is
        # a convenience, never the enforcement mechanism.
        processing_duration = min(longest_source, 180.0) if free_job else None
        await run_in_threadpool(prepare_vad_audio, dialogue_path, vad_path, processing_duration)
        try:
            speech_regions = await run_in_threadpool(detect_speech_samples, vad_path)
            logger.info("Audio job %s: VAD found %d speech regions", run_id, len(speech_regions))
            await run_in_threadpool(
                render_ducked_music, music_path, speech_regions, ducked_music_path, duck_vol
            )
        except AudioPipelineError:
            raise
        except Exception as exc:
            logger.exception("Audio job %s failed while analysing or ducking", run_id)
            raise AudioPipelineError("applying ducking curve", "Audio analysis failed.") from exc

        await run_in_threadpool(mix_audio, dialogue_path, ducked_music_path, output_path, free_job)

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AudioPipelineError("encoding output", "The encoder did not create an audio file.")

        job_id = uuid.uuid4().hex
        filename = "duckit-mix.mp3"
        expires_at = int(time.time()) + OUTPUT_TTL_SECONDS

        with db() as conn:
            if free_job:
                conn.execute(
                    "UPDATE identities SET free_exports_used = free_exports_used + 1 WHERE id = ?",
                    (identity_id,),
                )
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

        return {
            "job_id": job_id,
            "filename": filename,
            "download_url": f"/api/jobs/{job_id}/download",
            "exports_remaining": max(0, MAX_FREE_EXPORTS - exports_used),
            "free_exports_used": exports_used,
        }

    except HTTPException:
        output_path.unlink(missing_ok=True)
        raise
    except AudioPipelineError as exc:
        output_path.unlink(missing_ok=True)
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

    with db() as conn:
        seen = conn.execute("SELECT 1 FROM stripe_events WHERE id = ?", (event["id"],)).fetchone()
    if seen:
        return {"received": True}

    event_type = event["type"]
    obj = event["data"]["object"]

    if event_type in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
        subscription_id = stripe_id(obj.get("subscription"))
        if subscription_id:
            try:
                subscription = await run_in_threadpool(stripe.Subscription.retrieve, subscription_id)
                save_subscription(subscription, obj.get("client_reference_id"))
            except Exception:
                # Return a 5xx: Stripe will safely retry because the event has not been marked seen.
                logger.exception("Could not apply Stripe checkout event %s", event["id"])
                raise HTTPException(status_code=500, detail={"code": "webhook_processing_failed", "message": "Could not apply the Stripe subscription event."})
    elif event_type in {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}:
        try:
            save_subscription(obj)
        except Exception:
            logger.exception("Could not apply Stripe subscription event %s", event["id"])
            raise HTTPException(status_code=500, detail={"code": "webhook_processing_failed", "message": "Could not apply the Stripe subscription event."})

    with db() as conn:
        conn.execute("INSERT OR IGNORE INTO stripe_events (id, processed_at) VALUES (?, ?)", (event["id"], int(time.time())))

    return {"received": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
