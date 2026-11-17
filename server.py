import atexit
import asyncio
import logging
import os
import shutil
import sqlite3
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from posthog import Posthog
from posthog.integrations.asgi import PosthogASGIMiddleware
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

MAX_FREE_EXPORTS = 3
MAX_SOURCE_SECONDS = 60 * 60
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
OUTPUT_TTL_SECONDS = 24 * 60 * 60
COOKIE_NAME = "duckit_identity"
COOKIE_MAX_AGE = 365 * 24 * 60 * 60

APP_BASE_URL = os.environ.get("APP_BASE_URL", "").rstrip("/")
APP_SESSION_SECRET = os.environ.get("APP_SESSION_SECRET", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PRO_MONTHLY = os.environ.get("STRIPE_PRICE_PRO_MONTHLY", "")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    posthog_project_token: str = ""
    posthog_host: str = ""
    environment: str = "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
posthog_log_provider = None
posthog_log_logger = logging.getLogger("duckit.posthog_logs")
posthog_log_logger.setLevel(logging.INFO)
posthog_log_logger.propagate = False

if settings.posthog_project_token and settings.posthog_host:
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
if settings.posthog_project_token and settings.posthog_host:
    posthog_client = Posthog(
        settings.posthog_project_token,
        host=settings.posthog_host,
        enable_exception_autocapture=True,
    )
    atexit.register(posthog_client.shutdown)

if not APP_BASE_URL or not APP_SESSION_SECRET:
    raise RuntimeError("APP_BASE_URL and APP_SESSION_SECRET must be set.")

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

stripe.api_key = STRIPE_SECRET_KEY
signer = URLSafeTimedSerializer(APP_SESSION_SECRET, salt="duckit-identity-v1")

ml_models = {}

def load_vad_model_safely():
    torch.set_num_threads(1)
    try:
        logger.info("Loading local Silero VAD model from %s...", MODEL_PATH)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Local VAD model file not found at {MODEL_PATH}")
        
        model = torch.jit.load(str(MODEL_PATH))
        model.eval()
        ml_models["vad_model"] = model
        logger.info("Local Silero VAD model loaded successfully.")
    except Exception:
        logger.exception("Critical error: Failed to load local Silero VAD model.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vad_model_safely()
    yield
    if posthog_client:
        try:
            posthog_client.flush()
        except Exception:
            logger.exception("Could not flush PostHog events during shutdown")
    ml_models.clear()


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


def prepare_vad_audio(input_path: Path, output_path: Path):
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ar=16000, ac=1, format="wav")
        .run(overwrite_output=True, quiet=True)
    )


def apply_ducking_vectorized(music_audio, music_sr, speech_timestamps, duck_vol=0.15, fade_duration=0.05):
    envelope = np.ones(len(music_audio), dtype=np.float32)
    fade_samples = int(fade_duration * music_sr)
    scale = music_sr / 16000.0

    for ts in speech_timestamps:
        start = int(ts.get('start', 0) * scale)
        end = int(ts.get('end', 0) * scale)
        
        start = max(0, min(start, len(music_audio)))
        end = max(0, min(end, len(music_audio)))
        if start >= end:
            continue
        
        envelope[start:end] = duck_vol
        
        fade_out_start = max(0, start - fade_samples)
        if start > fade_out_start:
            ramp_out = np.linspace(1.0, duck_vol, start - fade_out_start)
            envelope[fade_out_start:start] = ramp_out
            
        fade_in_end = min(len(music_audio), end + fade_samples)
        if fade_in_end > end:
            ramp_in = np.linspace(duck_vol, 1.0, fade_in_end - end)
            envelope[end:fade_in_end] = ramp_in

    if music_audio.ndim > 1:
        envelope = envelope[:, np.newaxis]
        
    return music_audio * envelope


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health_check():
    return {"status": "online", "engine": "ready"}


@app.get("/api/billing/status")
def billing_status(identity_id: str = Depends(current_identity)):
    return entitlement_payload(identity_id)


@app.post("/api/billing/verify")
def verify_checkout_session(session_id: str = Form(...), identity_id: str = Depends(current_identity)):
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.get("payment_status") == "paid" or session.get("status") == "complete":
            subscription_id = session.get("subscription")
            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                save_subscription(subscription, identity_id)
    except Exception:
        logger.exception("Failed to verify checkout session")
    return entitlement_payload(identity_id)


@app.post("/api/billing/checkout")
def create_checkout(identity_id: str = Depends(current_identity)):
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_PRO_MONTHLY:
        raise HTTPException(status_code=503, detail="Billing is not configured.")

    user = user_row(identity_id)
    if is_pro(user):
        raise HTTPException(status_code=409, detail="Your Pro subscription is already active.")

    customer_id = user["stripe_customer_id"]
    if not customer_id:
        customer = stripe.Customer.create(metadata={"duckit_identity_id": identity_id})
        customer_id = customer.id
        with db() as conn:
            conn.execute(
                "UPDATE identities SET stripe_customer_id = ? WHERE id = ?",
                (customer_id, identity_id),
            )

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
    return {"url": checkout.url}


@app.post("/api/billing/portal")
def create_billing_portal(identity_id: str = Depends(current_identity)):
    user = user_row(identity_id)
    if not user or not user["stripe_customer_id"]:
        raise HTTPException(status_code=404, detail="No billing account exists yet.")

    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=APP_BASE_URL,
    )
    return {"url": portal.url}


@app.post("/api/duck", status_code=201)
async def duck_audio(
    dialogue: UploadFile = File(...),
    music: UploadFile = File(...),
    duck_vol: float = Form(0.15, ge=0.01, le=0.50),
    trim_seconds: int | None = Form(None),
    identity_id: str = Depends(current_identity),
):
    cleanup_expired_outputs()

    run_id = uuid.uuid4().hex
    dialogue_path = UPLOAD_DIR / f"{run_id}-dialogue.audio"
    music_path = UPLOAD_DIR / f"{run_id}-music.audio"
    vad_path = UPLOAD_DIR / f"{run_id}-vad.wav"
    ducked_music_path = UPLOAD_DIR / f"{run_id}-ducked-music.wav"
    output_path = OUTPUT_DIR / f"{run_id}.mp3"

    try:
        await save_upload(dialogue, dialogue_path)
        await save_upload(music, music_path)

        dialogue_seconds = source_duration(dialogue_path)
        music_seconds = source_duration(music_path)
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

        prepare_vad_audio(dialogue_path, vad_path)
        
        model = ml_models.get("vad_model")
        if not model:
            load_vad_model_safely()
            model = ml_models.get("vad_model")
            
        if not model:
            raise HTTPException(status_code=503, detail="VAD model failed to initialize.")

        data, _ = sf.read(vad_path)
        wav = torch.tensor(data, dtype=torch.float32)
        
        # Self-contained offline VAD chunk evaluation (bypasses missing helper dependencies)
        timestamps = []
        try:
            model.reset_states()
            chunk_size = 512
            audio_len = len(wav)
            in_speech = False
            start_sample = 0
            
            with torch.no_grad():
                for i in range(0, audio_len - chunk_size, chunk_size):
                    chunk = wav[i:i + chunk_size]
                    out = model(chunk, 16000)
                    prob = out.item() if hasattr(out, 'item') else float(out)
                    
                    if prob > 0.5 and not in_speech:
                        in_speech = True
                        start_sample = i
                    elif prob <= 0.5 and in_speech:
                        in_speech = False
                        end_sample = i
                        timestamps.append({"start": start_sample, "end": end_sample})
                if in_speech:
                    timestamps.append({"start": start_sample, "end": audio_len})
        except Exception:
            logger.exception("VAD chunk inference error, defaulting to full audio ducking envelope")
            
        if not timestamps:
            timestamps = [{"start": 0, "end": len(wav)}]

        music_data, music_sr = sf.read(str(music_path))
        ducked_music_array = apply_ducking_vectorized(
            music_data, music_sr, timestamps, duck_vol=duck_vol
        )
        sf.write(str(ducked_music_path), ducked_music_array, music_sr)

        input_dialogue = ffmpeg.input(str(dialogue_path))
        input_ducked_music = ffmpeg.input(str(ducked_music_path))

        mixed = ffmpeg.filter(
            [input_ducked_music, input_dialogue],
            "amix",
            inputs=2,
            duration="first",
            dropout_transition=0.5,
        )

        output_kwargs = {"acodec": "libmp3lame", "audio_bitrate": "320k"}
        if free_job and trim_seconds:
            output_kwargs["t"] = min(180, trim_seconds)

        ffmpeg.output(mixed, str(output_path), **output_kwargs).run(overwrite_output=True, quiet=True)

        if not output_path.exists():
            raise RuntimeError("Output file was not created.")

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
    except Exception:
        output_path.unlink(missing_ok=True)
        logger.exception("Audio processing failed")
        raise HTTPException(status_code=500, detail="Processing failed. Please try different audio files.")
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
    metadata = subscription.get("metadata", {})
    identity_id = identity_hint or metadata.get("duckit_identity_id")
    customer_id = subscription.get("customer")
    subscription_id = subscription.get("id")
    status = subscription.get("status", "none")

    with db() as conn:
        if identity_id:
            conn.execute(
                "UPDATE identities SET stripe_customer_id = ?, subscription_id = ?, subscription_status = ? WHERE id = ?",
                (customer_id, subscription_id, status, identity_id),
            )
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
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook is not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, signature, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    with db() as conn:
        seen = conn.execute("SELECT 1 FROM stripe_events WHERE id = ?", (event["id"],)).fetchone()
    if seen:
        return {"received": True}

    event_type = event["type"]
    obj = event["data"]["object"]

    if event_type in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
        subscription_id = obj.get("subscription")
        if subscription_id:
            subscription = stripe.Subscription.retrieve(subscription_id)
            save_subscription(subscription, obj.get("client_reference_id"))
    elif event_type in {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}:
        save_subscription(obj)

    with db() as conn:
        conn.execute("INSERT OR IGNORE INTO stripe_events (id, processed_at) VALUES (?, ?)", (event["id"], int(time.time())))

    return {"received": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
