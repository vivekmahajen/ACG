"""Stage 2 — Trend Analyser.

Feeds collected video data to Claude and receives a JSON object describing
the single best sub-topic to publish today.
"""

import json
import os
import time
from datetime import date

import anthropic

from utils import config as cfg
from utils import database as db
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior YouTube content strategist with 10 years of experience growing channels in competitive niches.

Your job is to analyse a list of recently-published YouTube videos and identify the single best sub-topic for a new 30-second short-form video to publish today.

Scoring criteria (in order of weight):
1. HIGH view count relative to channel average — this means the topic has strong demand
2. HIGH engagement rate (likes + comments / views) — this means viewers are emotionally activated
3. RECENCY — topics that appeared in videos published in the last 7 days are fresher than those from 30 days ago
4. UNIQUENESS — avoid topics already published by the channel in the last 7 days (provided in context)
5. DOMAIN FIT — the topic must be directly relevant to the specified domain focus

Do not suggest generic topics. Be specific. "How to save money" is too broad. "Why your coffee habit is costing you $1,200 a year" is specific.

Return ONLY valid JSON. No commentary. No markdown. No preamble."""

REQUIRED_OUTPUT_KEYS = {
    "topic", "why_this_topic", "hook", "key_visual_idea",
    "target_emotion", "estimated_watch_through_rate", "competitor_angle",
}

VALID_EMOTIONS = {"curiosity", "surprise", "urgency", "inspiration", "fear", "relief"}
VALID_WATCH_RATES = {"high", "medium", "low"}


def _build_user_prompt(niche: str, domain: str, videos: list[dict], recent_topics: list[str]) -> str:
    today = date.today().isoformat()
    video_lines = []
    for v in videos:
        video_lines.append(
            f"- Title: {v['title']} | Views: {v['view_count']:,} | "
            f"Likes: {v['like_count']:,} | Comments: {v['comment_count']:,} | "
            f"Engagement: {v['engagement_rate']:.3f} | Published: {v['published_at'][:10]} | "
            f"Tags: {', '.join(v.get('tags', [])[:5])}"
        )
    video_block = "\n".join(video_lines)

    dedup_block = (
        "\n".join(f"- {t}" for t in recent_topics)
        if recent_topics
        else "None — no recent videos published yet."
    )

    return f"""Niche: {niche}
Domain focus: {domain}
Today's date: {today}

Recently published videos to analyse:
{video_block}

Topics already published in the last 7 days (DO NOT repeat these):
{dedup_block}

Analyse the videos and call the submit_trend_analysis tool with your recommendation."""


def _extract_json(raw: str) -> dict:
    """Try multiple strategies to extract JSON from Claude's response."""
    raw = raw.strip()

    # Strategy 1: strip fences then parse
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find outermost { } and parse
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"invalid JSON: could not extract valid JSON from response.\nRaw: {raw[:600]}")


def _parse_and_validate(raw: str) -> dict:
    data = _extract_json(raw)

    missing = REQUIRED_OUTPUT_KEYS - set(data.keys())
    if missing:
        logger.warning("Stage 2 | Parsed keys: %s", list(data.keys()))
        raise ValueError(f"Claude response missing keys: {missing}")

    if len(data.get("topic", "")) > 120:
        raise ValueError(f"Topic too long ({len(data['topic'])} chars > 120)")

    emotion = data.get("target_emotion", "").lower()
    if emotion not in VALID_EMOTIONS:
        logger.warning("Unexpected emotion value: %r — keeping anyway", emotion)

    watch_rate = data.get("estimated_watch_through_rate", "").split()[0].lower()
    if watch_rate not in VALID_WATCH_RATES:
        logger.warning("Unexpected watch_through_rate: %r", data.get("estimated_watch_through_rate"))
    if watch_rate == "low":
        logger.warning("Stage 2 | estimated_watch_through_rate is LOW — proceeding anyway")

    return data


TREND_TOOL = {
    "name": "submit_trend_analysis",
    "description": "Submit the trend analysis result with the selected topic and all required fields.",
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Specific one-sentence topic for the video"},
            "why_this_topic": {"type": "string", "description": "Two sentences explaining the data signal"},
            "hook": {"type": "string", "description": "Exact first sentence of the video — under 10 words"},
            "key_visual_idea": {"type": "string", "description": "One sentence describing the strongest visual metaphor"},
            "target_emotion": {"type": "string", "enum": ["curiosity", "surprise", "urgency", "inspiration", "fear", "relief"]},
            "estimated_watch_through_rate": {"type": "string", "description": "high, medium, or low with one-sentence justification"},
            "competitor_angle": {"type": "string", "description": "How to differentiate from existing videos"},
        },
        "required": ["topic", "why_this_topic", "hook", "key_visual_idea", "target_emotion", "estimated_watch_through_rate", "competitor_angle"],
    },
}


def _call_claude(client: anthropic.Anthropic, model: str, user_prompt: str, attempt: int = 1) -> dict:
    # Use tool use to enforce the exact JSON schema — Claude must call the tool
    # with the required fields, making key name drift impossible.
    message = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        tools=[TREND_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Extract the tool input dict from the response
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_trend_analysis":
            data = block.input
            logger.debug("Stage 2 | Tool input: %s", str(data)[:600])
            return _parse_and_validate(json.dumps(data))
    raise ValueError(f"Claude did not call the trend tool. Response: {message.content}")


def run(stage1_output: dict, dry_run: bool = False) -> dict:
    """Execute Stage 2. Returns the trend analysis dict."""
    conf = cfg.load_config()
    niche: str = conf["niche"]
    domain: str = conf["domain"]
    model: str = conf["claude_model"]
    dedup_days: int = conf["dedup_lookback_days"]

    logger.info("Stage 2 | Analysing %d videos for niche=%s", len(stage1_output["videos"]), niche)

    recent_topics = db.get_recent_topics(dedup_days)
    logger.info("Stage 2 | Recent topics to avoid: %d", len(recent_topics))

    if dry_run:
        logger.info("Stage 2 | DRY-RUN — returning mock trend analysis")
        return _mock_stage2_output()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = _build_user_prompt(niche, domain, stage1_output["videos"], recent_topics)

    max_attempts = 2
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = _call_claude(client, model, user_prompt, attempt)
            logger.info("Stage 2 | Topic selected: %s", result["topic"])
            return result
        except ValueError as e:
            logger.warning("Stage 2 | Attempt %d/%d failed validation: %s", attempt, max_attempts, e)
            last_error = e
            if attempt < max_attempts:
                time.sleep(2)
        except anthropic.RateLimitError:
            logger.warning("Stage 2 | Rate limited — waiting 60s")
            time.sleep(60)
            last_error = anthropic.RateLimitError("rate limit", response=None, body=None)  # type: ignore[call-arg]

    raise RuntimeError(f"Stage 2 failed after {max_attempts} attempts: {last_error}") from last_error


def _mock_stage2_output() -> dict:
    return {
        "topic": "Why your daily coffee habit is silently costing you $1,200 every year",
        "why_this_topic": "The coffee-cost video from Millennial Money hit 980K views with a 5.7% engagement rate — the highest in the dataset. It was published 4 days ago so it is still trending.",
        "hook": "You're throwing away $1,200 a year and you don't even know it.",
        "key_visual_idea": "A close-up of coins spilling from a coffee cup, slow motion, warm golden tones.",
        "target_emotion": "surprise",
        "estimated_watch_through_rate": "high — the hook creates immediate financial anxiety that compels viewers to stay",
        "competitor_angle": "Competitors show the math in a spreadsheet. We show it visually — money literally dissolving into coffee steam.",
    }
