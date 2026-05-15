"""Stage 4b — Voiceover Generation.

Uses Claude to write a punchy spoken script for the video, then OpenAI TTS
to synthesise it, then ffmpeg to mix the audio onto the video MP4.
"""

import os
import time
from pathlib import Path

import anthropic
import requests

from utils import config as cfg
from utils.logger import get_logger

logger = get_logger(__name__)

VOICE = "nova"  # OpenAI TTS voice: alloy | echo | fable | onyx | nova | shimmer
TTS_MODEL = "tts-1"


SCRIPT_TOOL = {
    "name": "submit_voiceover_script",
    "description": "Submit the spoken voiceover script for the video.",
    "input_schema": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": (
                    "Spoken voiceover script. Must be 2–4 short punchy sentences. "
                    "No stage directions, no sound effects, just the words to speak aloud."
                ),
            }
        },
        "required": ["script"],
    },
}

SCRIPT_SYSTEM = """You write punchy voiceover scripts for 30-second YouTube Shorts.

Rules:
- Exactly 3 beats matching the 3 video scenes: Hook (0-10s) → Problem (10-20s) → Solution (20-30s)
- Each beat is 1–2 short sentences
- Total spoken length must fit in 28–30 seconds (roughly 70–85 words total)
- Hook beat: open with a surprising fact or question — make the viewer freeze
- Problem beat: amplify the cost or consequence with a specific number
- Solution beat: deliver the actionable insight, then close with exactly these words: "Thanks, Affordable Golden Years."
- Plain conversational English — no hashtags, no emojis, no markdown, no stage directions
- Do NOT describe visuals — this is audio only
- Write as one continuous script (no scene labels or headers)
- The final words of EVERY script must be: "Thanks, Affordable Golden Years." — no exceptions"""


def _generate_script(title: str, hook: str, topic: str, model: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = (
        f"Video topic: {topic}\n"
        f"YouTube title: {title}\n"
        f"Hook line: {hook}\n\n"
        "Write the voiceover script now."
    )

    for attempt in range(3):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=400,
                system=SCRIPT_SYSTEM,
                tools=[SCRIPT_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            for block in message.content:
                if block.type == "tool_use" and block.name == "submit_voiceover_script":
                    script = block.input.get("script", "").strip()
                    if len(script.split()) < 40:
                        raise ValueError(f"Script too short ({len(script.split())} words, min 40): {script!r}")
                    logger.info("Stage 4b | Script: %s", script)
                    return script
            raise ValueError("Claude did not call submit_voiceover_script")
        except ValueError as e:
            logger.warning("Stage 4b | Script attempt %d/3 failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(3)

    raise RuntimeError("Stage 4b | Failed to generate voiceover script after 3 attempts")


def _synthesise(script: str, audio_path: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set — cannot generate voiceover")

    logger.info("Stage 4b | Synthesising voiceover via OpenAI TTS")
    resp = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": TTS_MODEL, "voice": VOICE, "input": script, "response_format": "mp3"},
        timeout=30,
    )
    resp.raise_for_status()

    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(resp.content)
    logger.info("Stage 4b | Audio saved: %s (%d KB)", audio_path, len(resp.content) // 1024)
    return audio_path


def _mix(video_path: str, audio_path: str, output_path: str) -> str:
    import subprocess
    logger.info("Stage 4b | Mixing audio onto video")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mix failed: {result.stderr[-500:]}")
    logger.info("Stage 4b | Mixed video: %s", output_path)
    return output_path


def run(stage2_out: dict, stage3_out: dict, stage4_out: dict, dry_run: bool = False) -> dict:
    """Generate voiceover and mix onto video. Returns updated stage4_out with new video_file."""
    if dry_run:
        logger.info("Stage 4b | DRY-RUN — skipping audio generation")
        return stage4_out

    conf = cfg.load_config()
    model: str = conf["claude_model"]

    video_file: str = stage4_out["video_file"]
    title: str = stage3_out.get("title", "")
    hook: str = stage2_out.get("hook", "")
    topic: str = stage2_out.get("topic", "")

    # Paths
    stem = Path(video_file).stem
    audio_path = str(Path(video_file).parent / f"{stem}_vo.mp3")
    mixed_path = str(Path(video_file).parent / f"{stem}_audio.mp4")

    # Step 1: generate script
    script = _generate_script(title, hook, topic, model)

    # Step 2: TTS synthesis
    try:
        _synthesise(script, audio_path)
    except EnvironmentError as e:
        logger.warning("Stage 4b | %s — skipping audio", e)
        return stage4_out

    # Step 3: mix audio onto video
    try:
        mixed = _mix(video_file, audio_path, mixed_path)
        try:
            os.remove(audio_path)
        except OSError:
            pass
    except RuntimeError as e:
        logger.warning("Stage 4b | ffmpeg mix failed (%s) — using silent video", e)
        return stage4_out

    # Step 4: burn scrolling disclaimer ticker onto the video
    from utils import ffmpeg as _ffmpeg
    ticker_path = str(Path(video_file).parent / f"{stem}_final.mp4")
    try:
        _ffmpeg.add_ticker(mixed, ticker_path)
        try:
            os.remove(mixed)
        except OSError:
            pass
        logger.info("Stage 4b | Final video with ticker: %s", ticker_path)
        return {**stage4_out, "video_file": ticker_path, "has_audio": True}
    except Exception as e:
        logger.warning("Stage 4b | Ticker failed (%s) — using video without ticker", e)
        return {**stage4_out, "video_file": mixed, "has_audio": True}
