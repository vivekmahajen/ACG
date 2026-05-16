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

SCRIPT_SYSTEM = """You write punchy voiceover scripts for 50-second YouTube Shorts.

You will be given: the video topic, the problem hook, the specific actionable tip/solution, and a resource note.

Structure — 5 beats in strict order, matching the video timing exactly:

BEAT 1 — HOOK (seconds 0–3, ~8 words max)
One shocking number or counterintuitive claim. Make the viewer feel something is at stake right now.
Also tell them free resources are waiting in the pinned comment — weave it in naturally.
Example: "Most retirees overpay Medicare by $800 a year — resources are pinned below."

BEAT 2 — PROBLEM (seconds 3–12, ~20 words)
Explain why this problem affects the viewer specifically and personally right now. Make it feel urgent and real.

BEAT 3 — SOLUTION (seconds 12–35, ~45 words — the longest beat)
Deliver the single clear, actionable step with a concrete real-world example. Be specific — name the action, the amount, the outcome.
This is the core value of the video. Do not be vague. Do not rush it.

BEAT 4 — PROOF (seconds 35–45, ~20 words)
Give one specific number, statistic, or real result that makes the solution believable.
Something that makes the viewer think "that could be me."

BEAT 5 — CTA + SIGN-OFF (seconds 45–50)
Use EXACTLY this text, word for word:
"Subscribe, share, and comment for more money saving tips for tomorrow. Thanks, Affordable Golden Years."

Rules:
- Total spoken length: MUST fit inside 48 seconds — target roughly 105–115 words (NOT more)
- At natural speaking pace each word takes ~0.4 seconds — count your words before submitting
- Beat 3 (Solution) MUST be the longest beat — give it room
- The solution must directly match the tip provided — no bait-and-switch
- Plain conversational English — no hashtags, no emojis, no markdown, no stage directions
- Do NOT describe visuals — audio only
- Write as one continuous script with no labels or headers
- The final words of EVERY script must be exactly: "Subscribe, share, and comment for more money saving tips for tomorrow. Thanks, Affordable Golden Years." — no exceptions, no paraphrasing"""


def _build_resource_teaser(resources: list[dict]) -> str:
    return "free resources on this topic are linked in the pinned comment"


def _derive_tip(topic: str, solution_tip: str, competitor_angle: str, key_visual_idea: str) -> str:
    """Build the tip context for Claude — solution_tip is the primary source."""
    parts = []
    if solution_tip:
        parts.append(f"Actionable tip to tease and deliver: {solution_tip}")
    parts.append(f"Topic: {topic}")
    if competitor_angle:
        parts.append(f"Angle: {competitor_angle}")
    if key_visual_idea:
        parts.append(f"Key visual: {key_visual_idea}")
    return "\n".join(parts)


def _generate_script(title: str, hook: str, topic: str, model: str,
                     resources: list[dict] | None = None,
                     solution_tip: str = "",
                     competitor_angle: str = "",
                     key_visual_idea: str = "") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    resource_teaser = _build_resource_teaser(resources or [])
    tip_context = _derive_tip(topic, solution_tip, competitor_angle, key_visual_idea)

    user_prompt = (
        f"YouTube title: {title}\n"
        f"Problem hook (Beat 2): {hook}\n\n"
        f"Tip/solution context (use this to write Beat 1 teaser and Beat 4 delivery):\n{tip_context}\n\n"
        f"Resources teaser for Beat 4 mention: \"{resource_teaser}\"\n\n"
        "Write the 5-beat voiceover script now.\n"
        "Beat 1 must open with the shocking hook AND mention free resources are pinned below.\n"
        "Beat 3 (Solution) must be the longest beat — deliver the specific actionable tip with a real example.\n"
        "Beat 5 must be EXACTLY: 'Subscribe, share, and comment for more money saving tips for tomorrow. Thanks, Affordable Golden Years.'"
    )

    for attempt in range(3):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=500,
                system=SCRIPT_SYSTEM,
                tools=[SCRIPT_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            for block in message.content:
                if block.type == "tool_use" and block.name == "submit_voiceover_script":
                    script = block.input.get("script", "").strip()
                    if len(script.split()) < 100:
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


def _audio_duration(audio_path: str) -> float:
    """Return the duration of an audio file in seconds using ffprobe."""
    import subprocess, json
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", audio_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(json.loads(result.stdout)["format"]["duration"])


def _fit_audio_to_video(audio_path: str, video_duration: float) -> str:
    """Speed up the audio with atempo so it fits within video_duration.
    Returns the original path unchanged if it already fits."""
    import subprocess

    try:
        actual = _audio_duration(audio_path)
    except Exception as e:
        logger.warning("Stage 4b | Could not measure audio duration (%s) — skipping atempo", e)
        return audio_path

    logger.info("Stage 4b | Audio duration: %.1fs  Video duration: %.1fs", actual, video_duration)

    # Leave a 1-second buffer so the last word isn't cut at the very edge
    target = video_duration - 1.0
    if actual <= target:
        return audio_path

    ratio = actual / target
    logger.info("Stage 4b | Audio too long — speeding up by %.2fx with atempo", ratio)

    fitted_path = audio_path.replace(".mp3", "_fitted.mp3")
    # atempo accepts 0.5–2.0; chain two filters for ratios above 2.0
    if ratio <= 2.0:
        af = f"atempo={ratio:.4f}"
    else:
        af = f"atempo=2.0,atempo={ratio / 2:.4f}"

    cmd = ["ffmpeg", "-y", "-i", audio_path, "-filter:a", af, fitted_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Stage 4b | atempo failed (%s) — using original audio", result.stderr[-200:])
        return audio_path

    logger.info("Stage 4b | Audio fitted to %.1fs", target)
    return fitted_path


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


def run(stage2_out: dict, stage3_out: dict, stage4_out: dict,
        dry_run: bool = False, resources: list | None = None) -> dict:
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
    solution_tip: str = stage2_out.get("solution_tip", "")
    competitor_angle: str = stage2_out.get("competitor_angle", "")
    key_visual_idea: str = stage2_out.get("key_visual_idea", "")

    # Paths
    stem = Path(video_file).stem
    audio_path = str(Path(video_file).parent / f"{stem}_vo.mp3")
    mixed_path = str(Path(video_file).parent / f"{stem}_audio.mp4")

    # Step 1: generate script — tip is teased in Beat 1 and delivered in Beat 4
    script = _generate_script(
        title, hook, topic, model,
        resources=resources or [],
        solution_tip=solution_tip,
        competitor_angle=competitor_angle,
        key_visual_idea=key_visual_idea,
    )

    video_duration: float = float(stage4_out.get("video_duration", 50.0))

    # Step 2: TTS synthesis
    try:
        _synthesise(script, audio_path)
    except EnvironmentError as e:
        logger.warning("Stage 4b | %s — skipping audio", e)
        return stage4_out

    # Step 2b: fit audio to video length — speed up with atempo if over
    fitted_audio = _fit_audio_to_video(audio_path, video_duration)
    if fitted_audio != audio_path:
        try:
            os.remove(audio_path)
        except OSError:
            pass
        audio_path = fitted_audio

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
        import traceback
        logger.error("Stage 4b | Ticker failed — full error:\n%s", traceback.format_exc())
        return {**stage4_out, "video_file": mixed, "has_audio": True}
