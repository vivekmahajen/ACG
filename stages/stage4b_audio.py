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

SCRIPT_SYSTEM = """You write ultra-short, punchy voiceover scripts for 10-second YouTube Shorts.

Rules:
- 2 to 4 sentences maximum
- Each sentence is short and direct — no filler words
- Hook the viewer in the first sentence
- End with a curiosity gap or call to action
- Spoken aloud it must fit in under 10 seconds (roughly 25–35 words total)
- Plain conversational English — no hashtags, no emojis, no markdown
- Do NOT describe visuals — this is audio only"""


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
                max_tokens=200,
                system=SCRIPT_SYSTEM,
                tools=[SCRIPT_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            for block in message.content:
                if block.type == "tool_use" and block.name == "submit_voiceover_script":
                    script = block.input.get("script", "").strip()
                    if len(script.split()) < 10:
                        raise ValueError(f"Script too short: {script!r}")
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
        # Clean up intermediate audio file
        try:
            os.remove(audio_path)
        except OSError:
            pass
        return {**stage4_out, "video_file": mixed, "has_audio": True}
    except RuntimeError as e:
        logger.warning("Stage 4b | ffmpeg mix failed (%s) — using silent video", e)
        return stage4_out
