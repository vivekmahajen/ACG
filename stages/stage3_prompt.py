"""Stage 3 — Video Prompt & Metadata Generator.

Takes the approved trend topic and generates a cinematic text-to-video prompt
plus SEO-optimised YouTube upload metadata via Claude.
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

PART 2 — YOUTUBE SEO METADATA
Write fully SEO-optimised upload metadata:

- title: Under 60 characters. Lead with the primary keyword. Use a number or dollar amount if relevant. Create curiosity or urgency. Do NOT use clickbait that misrepresents the content.

- description: Exactly 3 lines structured for SEO:
  Line 1 (hook): Restate the core insight as a punchy statement that includes the primary keyword — this line appears before "Show more" so make it count.
  Line 2 (value): One sentence explaining what the viewer will learn or gain.
  Line 3 (CTA + hashtags): A follow/subscribe call to action followed by 5 hashtags: the 4 most relevant topic hashtags PLUS #Shorts at the end. Example: "Follow for daily money tips. #PersonalFinance #SavingMoney #MoneyHacks #MillennialMoney #Shorts"

- tags: Exactly 15 tags optimised for YouTube search. Mix:
  • 3 broad niche tags (e.g. "personal finance", "money tips", "financial advice")
  • 4 mid-tail topic tags (e.g. "how to save money fast", "money saving habits")
  • 4 long-tail exact-match tags (e.g. "why coffee is costing you money", "latte factor explained")
  • 2 audience tags (e.g. "personal finance for millennials", "money tips for beginners")
  • 2 format tags (e.g. "finance shorts", "money short video")

- hashtags: Exactly 5 hashtags (without the # symbol, as an array) — 4 topic-specific + "Shorts"

- thumbnail_concept: One sentence describing a high-contrast thumbnail image with a human face showing strong emotion, a bold number or dollar figure visible, and a clear single focal point. This is for AI image generation so be specific about colours and composition.

- pinned_comment: A single engaging question or statement to pin as the first comment. It must:
  • Invite viewers to reply with their own experience (drives comment velocity)
  • Naturally include 1-2 keywords
  • Be under 100 characters
  • Examples: "What's your biggest daily money waster? 👇" or "Comment your monthly coffee spend — I'll do the math 🧮"

- category: One of: Education / Entertainment / HowTo / News / Finance / Health / Technology / Lifestyle
- language: ISO 639-1 code e.g. "en"

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

REQUIRED_KEYS = {
    "video_prompt", "title", "description", "tags", "hashtags",
    "thumbnail_concept", "pinned_comment", "category", "language",
}


def _build_user_prompt(trend: dict, niche: str, domain: str) -> str:
    return f"""Niche: {niche}
Domain focus: {domain}
Trend topic: {trend['topic']}
Hook (first sentence of video): {trend['hook']}
Target emotion: {trend['target_emotion']}
Key visual idea: {trend['key_visual_idea']}
Competitor angle: {trend['competitor_angle']}

Generate the text-to-video prompt and SEO-optimised YouTube metadata JSON now."""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract valid JSON.\nRaw: {raw[:600]}")


def _validate(data: dict) -> None:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    prompt = data.get("video_prompt", "")
    if len(prompt.split()) < 100:
        raise ValueError(f"video_prompt too short: {len(prompt.split())} words (min 100)")

    if len(data.get("title", "")) > 60:
        raise ValueError(f"title too long: {len(data['title'])} chars (max 60)")

    if len(data.get("tags", [])) < 10:
        raise ValueError(f"Not enough tags: {len(data['tags'])} (min 10)")

    if len(data.get("hashtags", [])) < 3:
        raise ValueError(f"Not enough hashtags: {len(data['hashtags'])} (min 3)")

    if len(data.get("pinned_comment", "")) < 10:
        raise ValueError("pinned_comment too short")

    prompt_lower = prompt.lower()
    for word in DISQUALIFIED_WORDS:
        if word in prompt_lower:
            raise ValueError(f"video_prompt contains disqualified word: {repr(word)}")


def _enrich(data: dict) -> dict:
    """Post-process: ensure #Shorts is in hashtags and description, add category_id."""
    # Ensure Shorts tag is always present
    hashtags: list = data.get("hashtags", [])
    if "Shorts" not in hashtags and "shorts" not in [h.lower() for h in hashtags]:
        hashtags.append("Shorts")
    data["hashtags"] = hashtags

    # Ensure description ends with hashtags block
    desc: str = data.get("description", "")
    hashtag_block = " ".join(f"#{h}" for h in hashtags)
    if "#Shorts" not in desc and "#shorts" not in desc.lower():
        data["description"] = desc.rstrip() + f"\n{hashtag_block}"

    # Derive category_id
    data["category_id"] = CATEGORY_ID_MAP.get(data.get("category", "Finance"), "27")
    return data


def _call_claude(client: anthropic.Anthropic, model: str, user_prompt: str, attempt: int = 1) -> dict:
    strictness = "" if attempt == 1 else "\n\nIMPORTANT: Previous response failed validation. Return ONLY raw JSON."
    message = client.messages.create(
        model=model,
        max_tokens=1800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt + strictness}],
    )
    raw = message.content[0].text.strip()
    logger.debug("Stage 3 | Claude raw response:\n%s", raw[:800])

    data = _extract_json(raw)
    _validate(data)
    return _enrich(data)


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
            logger.info("Stage 3 | Tags: %d | Hashtags: %d", len(result["tags"]), len(result["hashtags"]))
            logger.info("Stage 3 | Pinned comment: %s", result["pinned_comment"])
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
            "Your daily coffee is quietly draining $1,200 from your savings every single year.\n"
            "Here's the exact math that will make you rethink your morning routine.\n"
            "Follow for daily money tips that actually move the needle. "
            "#PersonalFinance #SavingMoney #MoneyHacks #MillennialMoney #Shorts"
        ),
        "tags": [
            "personal finance", "money tips", "financial advice",
            "how to save money fast", "money saving habits", "latte factor", "cut daily expenses",
            "why coffee is costing you money", "latte factor explained", "save $1200 a year", "daily spending audit",
            "personal finance for millennials", "money tips for beginners",
            "finance shorts", "money short video",
        ],
        "hashtags": ["PersonalFinance", "SavingMoney", "MoneyHacks", "MillennialMoney", "Shorts"],
        "thumbnail_concept": (
            "Close-up of a shocked young woman's face (wide eyes, open mouth) on the left, "
            "with a bold red '$1,200' in large white text on a dark background on the right, "
            "a coffee cup icon beneath it. High contrast, warm orange accent colour."
        ),
        "pinned_comment": "What's your biggest daily money waster? Drop it below 👇",
        "category": "Finance",
        "language": "en",
        "category_id": "27",
    }
