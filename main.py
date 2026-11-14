import os
import sys
import argparse
import torch
import ffmpeg
import soundfile as sf

def prepare_vad_audio(input_path, output_path):
    """Creates a temporary 16kHz mono WAV strictly for Silero VAD scanning."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ar=16000, ac=1, format='wav')
            .run(overwrite_output=True, quiet=True)
        )
    except ffmpeg.Error as e:
        print(f"[-] FFmpeg error during VAD preparation: {e.stderr.decode('utf8')}", file=sys.stderr)
        sys.exit(1)

def build_smooth_ducking_expr(timestamps, duck_vol=0.15, fade_sec=0.25, pad_sec=0.15):
    """Generates an FFmpeg volume expression with smooth fade-in/fade-out ramps."""
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

def run_duckit(dialogue_path, music_path, output_path, duck_vol):
    if not os.path.exists(dialogue_path):
        print(f"[-] Error: Dialogue file not found -> {dialogue_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(music_path):
        print(f"[-] Error: Music file not found -> {music_path}", file=sys.stderr)
        sys.exit(1)

    vad_temp = "assets/temp_vad.wav"
    print(f"[*] Preparing dialogue: {dialogue_path}")
    prepare_vad_audio(dialogue_path, vad_temp)

    print("[*] Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils

    print("[*] Scanning dialogue for voice activity...")
    data, _ = sf.read(vad_temp)
    wav = torch.tensor(data, dtype=torch.float32)
    
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    print(f"[*] Found {len(speech_timestamps)} speech segments!")

    vol_filter = build_smooth_ducking_expr(speech_timestamps, duck_vol=duck_vol, fade_sec=0.25, pad_sec=0.15)

    print(f"[*] Ducking music (level: {int(duck_vol*100)}%) and mixing tracks...")
    input_dialogue = ffmpeg.input(dialogue_path)
    input_music = ffmpeg.input(music_path).filter('volume', volume=vol_filter, eval='frame')
    
    mixed = ffmpeg.filter([input_music, input_dialogue], 'amix', inputs=2, duration='first', dropout_transition=0.5)
    ffmpeg.output(mixed, output_path, audio_bitrate='320k').run(overwrite_output=True, quiet=True)
    
    if os.path.exists(vad_temp):
        os.remove(vad_temp)
        
    print(f"[🔥] DONE! Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckIt: Automated AI Audio Ducking CLI")
    parser.add_argument("dialogue", help="Path to voice/dialogue audio file")
    parser.add_argument("music", help="Path to background music/beat audio file")
    parser.add_argument("-o", "--output", default="assets/output_ducked.mp3", help="Path for output file")
    parser.add_argument("-d", "--duck", type=float, default=0.15, help="Ducked music volume level (0.0 to 1.0, default: 0.15)")
    
    args = parser.parse_args()
    run_duckit(args.dialogue, args.music, args.output, args.duck)
