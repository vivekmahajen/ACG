"""Stage 3 — Video Prompt & Metadata Generator.

Takes the approved trend topic and generates a cinematic text-to-video prompt
plus complete YouTube upload metadata via Claude.
"""

import json
import os
import time

import anthropic

from utils import config as cfg
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are simultaneously a film director, a YouTube SEO specialist, and a viral short-form content strategist.

You will receive a trend topic and must produce two things in a single response:

PART 1 — TEXT-TO-VIDEO PROMPT
Write a cinematic, detailed prompt for a text-to-video AI model (Kling or Runway). The prompt must:
- Be between 120 and 180 words
- Describe the visuals in present tense ("A woman sits at a desk...")
- Specify camera movement (slow zoom in / tracking shot / handheld / aerial / static)
- Specify lighting (golden hour / studio softbox / neon reflections / natural window light)
- Specify mood and colour grade (warm and cosy / cold and clinical / vibrant and energetic)
- Specify the visual metaphor or central image that carries the topic
- Be written as a single flowing paragraph with no bullet points
- NOT include any text overlays, subtitles, or typography in the visual — the AI cannot render text reliably
- Be optimised for a 9:16 vertical frame (phone screen ratio)
- Feel premium and cinematic — not stock footage aesthetic

PART 2 — YOUTUBE METADATA
Write complete upload metadata:
- Title: under 60 characters, includes the primary keyword naturally, creates curiosity or urgency
- Description: 3 sentences — one hook, one value statement, one call to action
- Tags: exactly 12 tags, mix of broad (niche) and specific (topic) keywords
- Thumbnail concept: one sentence describing what the thumbnail image should show (for later manual or AI generation)

Return ONLY valid JSON. No commentary. No markdown fences."""

DISQUALIFIED_WORDS = {"text", "subtitle", "caption", "overlay", "title card", "words appear"}

CATEGORY_ID_MAP = {
    "Education": "27",
    "Entertainment": "24",
    "HowTo": "26",
    "News": "25",
    "Finance": "27",
    "Health": "26",
    "Technology": "28",
    "Lifestyle": "22",
}

REQUIRED_KEYS = {"video_prompt", "title", "description", "tags", "thumbnail_concept", "category", "language"}


def _build_user_prompt(trend: dict, niche: str, domain: str) -> str:
    return f"""Niche: {niche}
Domain focus: {domain}
Trend topic: {trend['topic']}
Hook (first sentence of video): {trend['hook']}
Target emotion: {trend['target_emotion']}
Key visual idea: {trend['key_visual_idea']}
Competitor angle: {trend['competitor_angle']}

Generate the text-to-video prompt and YouTube metadata JSON now."""


def _validate(data: dict) -> None:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    prompt = data.get("video_prompt", "")
    word_count = len(prompt.split())
    if word_count < 100:
        raise ValueError(f"video_prompt too short: {word_count} words (minimum 100)")

    title = data.get("title", "")
    if len(title) > 60:
        raise ValueError(f"title too long: {len(title)} chars (max 60)")

    tags = data.get("tags", [])
    if len(tags) < 8:
        raise ValueError(f"Not enough tags: {len(tags)} (minimum 8)")

    # Hard disqualification — text rendering
    prompt_lower = prompt.lower()
    for word in DISQUALIFIED_WORDS:
        if word in prompt_lower:
            raise ValueError(f"video_prompt contains disqualified word: {repr(word)}")


def _call_claude(client: anthropic.Anthropic, model: str, user_prompt: str, attempt: int = 1) -> dict:
    strictness = "" if attempt == 1 else "\n\nIMPORTANT: Previous response failed validation. Return ONLY raw JSON."
    message = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt + strictness}],
    )
    raw = message.content[0].text.strip()
    logger.debug("Stage 3 | Claude raw response:\n%s", raw[:800])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nRaw: {raw[:500]}") from e

    _validate(data)

    # Attach derived category_id
    cat = data.get("category", "Education")
    data["category_id"] = CATEGORY_ID_MAP.get(cat, "27")

    return data


def run(stage2_output: dict, dry_run: bool = False) -> dict:
    """Execute Stage 3. Returns prompt and metadata dict."""
    conf = cfg.load_config()
    niche: str = conf["niche"]
    domain: str = conf["domain"]
    model: str = conf["claude_model"]

    logger.info("Stage 3 | Generating video prompt for topic: %s", stage2_output["topic"])

    if dry_run:
        logger.info("Stage 3 | DRY-RUN — returning mock prompt and metadata")
        return _mock_stage3_output()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = _build_user_prompt(stage2_output, niche, domain)

    max_attempts = 3
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = _call_claude(client, model, user_prompt, attempt)
            logger.info("Stage 3 | Title: %s", result["title"])
            logger.info("Stage 3 | Prompt word count: %d", len(result["video_prompt"].split()))
            return result
        except ValueError as e:
            logger.warning("Stage 3 | Attempt %d/%d failed: %s", attempt, max_attempts, e)
            last_error = e
            if attempt < max_attempts:
                time.sleep(3)
        except anthropic.RateLimitError:
            logger.warning("Stage 3 | Rate limited — waiting 60s")
            time.sleep(60)
            last_error = Exception("rate limit")

    raise RuntimeError(f"Stage 3 failed after {max_attempts} attempts: {last_error}") from last_error


def _mock_stage3_output() -> dict:
    return {
        "video_prompt": (
            "A young woman sits alone at a cosy kitchen table in warm golden morning light, "
            "her hands wrapped around a steaming ceramic coffee mug. The camera begins in a "
            "tight close-up on the mug and slowly pulls back in a smooth tracking shot, "
            "revealing a growing pile of coins and crumpled dollar bills scattered across "
            "the wooden table around her. Her expression shifts from relaxed to quietly stunned "
            "as she glances down at the money. The colour grade is warm amber with soft shadows, "
            "evoking comfort turning into realisation. A single shaft of natural window light "
            "falls diagonally across the scene, making the coins glint. The 9:16 vertical frame "
            "keeps her face and the money pile centred, creating an intimate confessional mood. "
            "The overall aesthetic is cinematic and premium — warm, human, and emotionally resonant."
        ),
        "title": "Your coffee habit costs $1,200/year ☕",
        "description": (
            "You make this purchase every single day without thinking — but it's silently draining your savings. "
            "We did the maths and the number will shock you. "
            "Follow for more money moves that actually work."
        ),
        "tags": [
            "personal finance", "saving money", "coffee latte factor", "money tips",
            "millennial money", "budgeting", "financial freedom", "frugal living",
            "money habits", "save more money", "daily expenses", "financial tips",
        ],
        "thumbnail_concept": "Close-up of coins overflowing from a coffee cup on a wooden table, warm morning light.",
        "category": "Finance",
        "language": "en",
        "category_id": "27",
    }
