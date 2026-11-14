import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import ffmpeg
import soundfile as sf

app = FastAPI(title="DuckIt API", description="Automated AI Audio Ducking Micro-SaaS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_vad_audio(input_path, output_path):
    (
        ffmpeg
        .input(input_path)
        .output(output_path, ar=16000, ac=1, format='wav')
        .run(overwrite_output=True, quiet=True)
    )

def build_smooth_ducking_expr(timestamps, duck_vol=0.15, fade_sec=0.25, pad_sec=0.15):
    if not timestamps:
        return "1.0"
    
    padded = []
    for ts in timestamps:
        start = max(0.0, (ts['start'] / 16000.0) - pad_sec)
        end = (ts['end'] / 16000.0) + pad_sec
        if padded and start <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], end)
        else:
            padded.append([start, end])
            
    drop = 1.0 - duck_vol
    expr_parts = ["1.0"]
    
    for start, end in padded:
        fade_in_start = max(0.0, start - fade_sec)
        fade_out_end = end + fade_sec
        
        ramp_down = f"if(between(t,{fade_in_start:.3f},{start:.3f}), {drop:.3f}*((t-{fade_in_start:.3f})/{fade_sec:.3f}), 0)"
        hold = f"if(between(t,{start:.3f},{end:.3f}), {drop:.3f}, 0)"
        ramp_up = f"if(between(t,{end:.3f},{fade_out_end:.3f}), {drop:.3f}*(1-((t-{end:.3f})/{fade_sec:.3f})), 0)"
        
        expr_parts.append(f"-({ramp_down}+{hold}+{ramp_up})")
        
    return "".join(expr_parts)

@app.post("/api/duck")
async def duck_audio(
    dialogue: UploadFile = File(...),
    music: UploadFile = File(...),
    duck_vol: float = Form(0.15)
):
    dialogue_path = os.path.join(UPLOAD_DIR, dialogue.filename)
    music_path = os.path.join(UPLOAD_DIR, music.filename)
    output_filename = f"ducked_{os.path.splitext(dialogue.filename)[0]}.mp3"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    vad_temp = os.path.join(UPLOAD_DIR, "temp_vad.wav")

    with open(dialogue_path, "wb") as buffer:
        shutil.copyfileobj(dialogue.file, buffer)
    with open(music_path, "wb") as buffer:
        shutil.copyfileobj(music.file, buffer)

   try:
        prepare_vad_audio(dialogue_path, vad_temp)

        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, _, _, _, _) = utils

        data, _ = sf.read(vad_temp)
        wav = torch.tensor(data, dtype=torch.float32)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

        vol_filter = build_smooth_ducking_expr(speech_timestamps, duck_vol=duck_vol)

        input_dialogue = ffmpeg.input(dialogue_path)
        input_music = ffmpeg.input(music_path).filter('volume', volume=vol_filter, eval='frame')
        
        mixed = ffmpeg.filter([input_music, input_dialogue], 'amix', inputs=2, duration='first', dropout_transition=0.5)
        ffmpeg.output(mixed, output_path, audio_bitrate='320k').run(overwrite_output=True, quiet=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(vad_temp):
            os.remove(vad_temp)

    return FileResponse(output_path, media_type="audio/mpeg", filename=output_filename)

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
