import asyncio
import logging
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("duckit")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
DATA_DIR = BASE_DIR / ".duckit-data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "private-outputs"
DB_PATH = DATA_DIR / "duckit.sqlite3"

FREE_LIMIT_SECONDS = 120
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

if not APP_BASE_URL or not APP_SESSION_SECRET:
    raise RuntimeError("APP_BASE_URL and APP_SESSION_SECRET must be set.")

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

stripe.api_key = STRIPE_SECRET_KEY
signer = URLSafeTimedSerializer(APP_SESSION_SECRET, salt="duckit-identity-v1")

app = FastAPI(title="DuckIt API", description="Automated AI Audio Ducking")

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

    if identity_id:
        return identity_id

    identity_id = uuid.uuid4().hex
    with db() as conn:
        conn.execute(
            "INSERT INTO identities (id, created_at) VALUES (?, ?)",
            (identity_id, int(time.time())),
        )
    set_identity_cookie(response, identity_id)
    return identity_id


def entitlement_payload(identity_id: str):
    user = user_row(identity_id)
    pro = is_pro(user)
    return {
        "plan": "pro" if pro else "free",
        "pro": pro,
        "free_limit_seconds": FREE_LIMIT_SECONDS,
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


@lru_cache(maxsize=1)
def vad_components():
    torch.set_num_threads(1)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils[0]


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(INDEX_FILE, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health_check():
    return {"status": "online", "engine": "ready"}


@app.get("/api/billing/status")
def billing_status(identity_id: str = Depends(current_identity)):
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
        customer = stripe.Customer.create(
            metadata={"duckit_identity_id": identity_id}
        )
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
        subscription_data={
            "metadata": {"duckit_identity_id": identity_id}
        },
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
    identity_id: str = Depends(current_identity),
):
    cleanup_expired_outputs()

    run_id = uuid.uuid4().hex
    dialogue_path = UPLOAD_DIR / f"{run_id}-dialogue.audio"
    music_path = UPLOAD_DIR / f"{run_id}-music.audio"
    vad_path = UPLOAD_DIR / f"{run_id}-vad.wav"
    sidechain_wav_path = UPLOAD_DIR / f"{run_id}-sidechain.wav"
    output_path = OUTPUT_DIR / f"{run_id}.mp3"

    try:
        await save_upload(dialogue, dialogue_path)
        await save_upload(music, music_path)

        dialogue_seconds = source_duration(dialogue_path)
        music_seconds = source_duration(music_path)
        longest_source = max(dialogue_seconds, music_seconds)
        user = user_row(identity_id)
        free_job = not is_pro(user)

        if longest_source > MAX_SOURCE_SECONDS:
            raise HTTPException(
                status_code=413,
                detail="Source tracks must be one hour or shorter.",
            )

        if free_job and longest_source > FREE_LIMIT_SECONDS:
            raise HTTPException(
                status_code=402,
                detail={
                    "code": "upgrade_required",
                    "message": "Free exports support tracks up to 2 minutes. Upgrade to Pro for unlimited exports.",
                },
            )

        prepare_vad_audio(dialogue_path, vad_path)
        model, get_speech_timestamps = vad_components()

        data, _ = sf.read(vad_path)
        wav = torch.tensor(data, dtype=torch.float32)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

        sidechain_data = np.zeros_like(data)
        for ts in timestamps:
            start_idx = ts["start"]
            end_idx = ts["end"]
            sidechain_data[start_idx:end_idx] = data[start_idx:end_idx]

        sf.write(str(sidechain_wav_path), sidechain_data, 16000)

        input_dialogue = ffmpeg.input(str(dialogue_path))
        input_music = ffmpeg.input(str(music_path))
        input_sidechain = ffmpeg.input(str(sidechain_wav_path))

        # Smoothed sidechain compression with gradual attack and long release to eliminate choppy pumping
        ducked_music = ffmpeg.filter(
            [input_music, input_sidechain],
            "sidechaincompress",
            threshold=0.015,
            ratio=4.0,
            attack=50,
            release=800,
            makeup=1.0,
        )

        mixed = ffmpeg.filter(
            [ducked_music, input_dialogue],
            "amix",
            inputs=2,
            duration="first",
            dropout_transition=0.5,
        )

        ffmpeg.output(
            mixed,
            str(output_path),
            acodec="libmp3lame",
            audio_bitrate="320k",
        ).run(overwrite_output=True, quiet=True)

        if not output_path.exists():
            raise RuntimeError("Output file was not created.")

        job_id = uuid.uuid4().hex
        filename = "duckit-mix.mp3"
        expires_at = int(time.time()) + OUTPUT_TTL_SECONDS

        with db() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                (id, identity_id, output_path, filename, is_free, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    identity_id,
                    str(output_path),
                    filename,
                    int(free_job),
                    expires_at,
                    int(time.time()),
                ),
            )

        return {
            "job_id": job_id,
            "filename": filename,
            "download_url": f"/api/jobs/{job_id}/download",
        }

    except HTTPException:
        output_path.unlink(missing_ok=True)
        raise
    except Exception:
        output_path.unlink(missing_ok=True)
        logger.exception("Audio processing failed")
        raise HTTPException(
            status_code=500,
            detail="Processing failed. Please try different audio files.",
        )
    finally:
        for path in (dialogue_path, music_path, vad_path, sidechain_wav_path):
            path.unlink(missing_ok=True)


@app.get("/api/jobs/{job_id}/download")
def download_job(
    job_id: str,
    response: Response,
    identity_id: str = Depends(current_identity),
):
    with db() as conn:
        job = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND identity_id = ?",
            (job_id, identity_id),
        ).fetchone()

    if not job:
        raise HTTPException(status_code=404, detail="Output not found.")

    if job["expires_at"] < int(time.time()) or not Path(job["output_path"]).exists():
        raise HTTPException(status_code=410, detail="This output has expired. Please process it again.")

    if not job["is_free"] and not is_pro(user_row(identity_id)):
        raise HTTPException(
            status_code=402,
            detail={
                "code": "upgrade_required",
                "message": "An active Pro subscription is required to download this export.",
            },
        )

    return FileResponse(
        job["output_path"],
        media_type="audio/mpeg",
        filename=job["filename"],
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def save_subscription(subscription, identity_hint=None):
    metadata = subscription.get("metadata", {})
    identity_id = identity_hint or metadata.get("duckit_identity_id")
    customer_id = subscription.get("customer")
    subscription_id = subscription.get("id")
    status = subscription.get("status", "none")

    with db() as conn:
        if identity_id:
            conn.execute(
                """
                UPDATE identities
                SET stripe_customer_id = ?, subscription_id = ?, subscription_status = ?
                WHERE id = ?
                """,
                (customer_id, subscription_id, status, identity_id),
            )
        else:
            conn.execute(
                """
                UPDATE identities
                SET subscription_id = ?, subscription_status = ?
                WHERE stripe_customer_id = ? OR subscription_id = ?
                """,
                (subscription_id, status, customer_id, subscription_id),
            )


@app.post("/api/stripe/webhook", include_in_schema=False)
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook is not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, signature, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid webhook payload.")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    with db() as conn:
        seen = conn.execute(
            "SELECT 1 FROM stripe_events WHERE id = ?",
            (event["id"],),
        ).fetchone()
    if seen:
        return {"received": True}

    event_type = event["type"]
    obj = event["data"]["object"]

    if event_type in {
        "checkout.session.completed",
        "checkout.session.async_payment_succeeded",
    }:
        subscription_id = obj.get("subscription")
        if subscription_id:
            subscription = stripe.Subscription.retrieve(subscription_id)
            save_subscription(subscription, obj.get("client_reference_id"))

    elif event_type in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }:
        save_subscription(obj)

    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO stripe_events (id, processed_at) VALUES (?, ?)",
            (event["id"], int(time.time())),
        )

    return {"received": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
