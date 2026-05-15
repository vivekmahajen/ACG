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

You will be given: the video topic, the problem hook, the specific tip/solution, and a resource teaser.

Structure — 4 beats in strict order:

BEAT 1 — RETENTION HOOK (first 3–4 seconds, ~18 words max)
Two parts in one breath:
1. Tease the SPECIFIC TIP that will be revealed at the end — make the viewer feel they will miss something valuable if they leave.
2. In the same sentence or immediately after, tell them free resources are waiting in the pinned comment.
Example: "Stay to the end — I'll show you the one switch that stops this drain, and I've linked free resources in the pinned comment."
Use the tip provided. Vary the wording every time. Never start two videos the same way.

BEAT 2 — PROBLEM HOOK (seconds 3–10)
Hit them with the surprising fact or number. Make the viewer feel the problem is happening to them right now.

BEAT 3 — AMPLIFICATION (seconds 10–20)
Deepen the cost or consequence with a specific dollar figure or statistic. Make the scale land emotionally.

BEAT 4 — TIP DELIVERY + ENGAGEMENT CTA + SIGN-OFF (seconds 20–30)
This beat has three parts — all must appear, in this order:

PART A — TIP DELIVERY
Deliver the tip from Beat 1 explicitly and completely. This is the payoff the viewer stayed for.
The tip must directly resolve what was teased in Beat 1 — no bait-and-switch.

PART B — ENGAGEMENT CTA (cover ALL of these signals naturally in 2–3 sentences)
• Like: ask them to hit like if it helped
• Comment: ask a specific question tied to the topic so they reply (e.g. "How much do you spend on coffee monthly? Drop it below.")
• Share: ask them to share with family or friends who need this
• Subscribe/Follow: ask them to follow for daily tips
• Save: ask them to save the video so they can come back to it
• Resources: mention free resources are pinned in the comments
Do NOT list these as bullet points — weave them into natural spoken sentences.

PART C — SIGN-OFF
Close with exactly: "Thanks, Affordable Golden Years."

Rules:
- Total spoken length: 33–40 seconds (roughly 95–115 words) — Beat 1 and the CTA both need room
- The tip in Part A MUST directly answer what Beat 1 teased
- Plain conversational English — no hashtags, no emojis, no markdown, no stage directions
- Do NOT describe visuals — audio only
- Write as one continuous script with no labels or headers
- The final words of EVERY script must be: "Thanks, Affordable Golden Years." — no exceptions"""


def _build_resource_teaser(resources: list[dict]) -> str:
    if not resources:
        return "free official resources that can help you take action today"
    count = len(resources)
    domains = []
    for r in resources[:3]:
        url = r.get("url", "")
        import re
        m = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if m:
            domains.append(m.group(1))
    if domains:
        return f"{count} free resources including {' and '.join(domains[:2])}"
    return f"{count} free official resources"


def _derive_tip(topic: str, competitor_angle: str, key_visual_idea: str) -> str:
    """Build a concise tip description for Claude to tease and deliver."""
    parts = [f"Topic: {topic}"]
    if competitor_angle:
        parts.append(f"Solution angle: {competitor_angle}")
    if key_visual_idea:
        parts.append(f"Key visual: {key_visual_idea}")
    return "\n".join(parts)


def _generate_script(title: str, hook: str, topic: str, model: str,
                     resources: list[dict] | None = None,
                     competitor_angle: str = "",
                     key_visual_idea: str = "") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    resource_teaser = _build_resource_teaser(resources or [])
    tip_context = _derive_tip(topic, competitor_angle, key_visual_idea)

    user_prompt = (
        f"YouTube title: {title}\n"
        f"Problem hook (Beat 2): {hook}\n\n"
        f"Tip/solution context (use this to write Beat 1 teaser and Beat 4 delivery):\n{tip_context}\n\n"
        f"Resources teaser for Beat 4 mention: \"{resource_teaser}\"\n\n"
        "Write the 4-beat voiceover script now.\n"
        "Beat 1 must tease the specific tip AND tell viewers free resources are in the pinned comment.\n"
        "Beat 4 must deliver that tip explicitly, remind viewers the resources are pinned, then sign off."
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
        competitor_angle=competitor_angle,
        key_visual_idea=key_visual_idea,
    )

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
