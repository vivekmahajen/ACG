"""Stage 3 — Video Prompt & Metadata Generator.

Takes the approved trend topic and generates three cinematic 10-second scene
prompts (forming a 30-second Short) plus SEO-optimised YouTube upload metadata.
"""

import os
import time

import anthropic

from utils import config as cfg
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are simultaneously a film director, a YouTube SEO specialist, and a viral short-form content strategist.

You will receive a trend topic and must produce two things in a single response:

PART 1 — THREE SCENE PROMPTS FOR A 30-SECOND SHORT
Write three sequential cinematic prompts for a text-to-video AI model (Kling). Each scene is exactly 10 seconds. Together they form a complete 30-second story arc.

FACIAL EXPRESSION ARC — THIS IS MANDATORY IN EVERY VIDEO:
The subject's face must follow this exact emotional journey across the three scenes:
- Scene 1: expression is serious, concerned, or mildly worried — they sense something is wrong
- Scene 2: expression deepens to troubled, grim, or visibly stressed — the problem is hitting them
- Scene 3: expression gradually softens and opens into a warm, relieved smile — the solution has landed and they feel grateful and hopeful
Every scene prompt MUST explicitly describe the subject's facial expression following this arc. A flat or neutral face is not acceptable.

SCENE 1 — HOOK (seconds 0–10)
Grab attention immediately. Open mid-action. Establish the central problem or surprising fact visually. Make the viewer feel something is at stake. Subject's face is serious and concerned.

SCENE 2 — AMPLIFICATION (seconds 10–20)
Deepen the impact. Show the scale, cost, or consequence. Make the emotional weight land. Use a visual that makes the number or problem feel real and personal. Subject's face is grim or visibly stressed.

SCENE 3 — RESOLUTION (seconds 20–30)
Deliver the insight, solution, or curiosity gap. End on a visual that implies positive change or drives the viewer to comment or follow. Subject's face transitions from serious to a warm, genuine smile — the kind of smile that says "I've got this now."

Each scene prompt must:
- Be between 80 and 120 words
- Describe the visuals in present tense ("A woman sits at a desk...")
- Explicitly describe the subject's facial expression (see arc above)
- Specify camera movement (slow zoom in / tracking shot / handheld / aerial / static)
- Specify lighting (golden hour / studio softbox / neon reflections / natural window light)
- Specify mood and colour grade (warm and cosy / cold and clinical / vibrant and energetic)
- Be a single flowing paragraph with no bullet points
- NOT include any text overlays, subtitles, captions, or typography — the AI cannot render text reliably
- Be optimised for a 9:16 vertical frame (phone screen ratio)
- Feel premium and cinematic — not stock footage aesthetic
- Be consistent with the other scenes in character, setting, and colour grade where possible

PART 2 — YOUTUBE SEO METADATA
Write fully SEO-optimised upload metadata:

- title: Under 60 characters. Lead with the primary keyword. Use a number or dollar amount if relevant. Create curiosity or urgency. Do NOT use clickbait that misrepresents the content.

- description: Exactly 3 lines structured for SEO:
  Line 1 (hook): Restate the core insight as a punchy statement that includes the primary keyword — this line appears before "Show more" so make it count.
  Line 2 (value): One sentence explaining what the viewer will learn or gain.
  Line 3 (CTA + hashtags): A follow/subscribe call to action followed by 5 hashtags: the 4 most relevant topic hashtags PLUS #Shorts at the end.

- tags: Exactly 15 tags optimised for YouTube search. Mix:
  • 3 broad niche tags
  • 4 mid-tail topic tags
  • 4 long-tail exact-match tags
  • 2 audience tags
  • 2 format tags

- hashtags: Exactly 5 hashtags (without the # symbol, as an array) — 4 topic-specific + "Shorts"

- thumbnail_concept: One sentence describing a high-contrast thumbnail image with a human face showing strong emotion, a bold number or dollar figure visible, and a clear single focal point. Specific about colours and composition — this is for AI image generation.

- pinned_comment: A single engaging question or statement under 100 characters that invites viewers to reply with their own experience and includes 1-2 keywords.

- category: One of: Education / Entertainment / HowTo / News / Finance / Health / Technology / Lifestyle
- language: ISO 639-1 code e.g. "en" """

DISQUALIFIED_WORDS = {"subtitle", "caption", "overlay", "title card", "words appear"}

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
    "video_prompt_1", "video_prompt_2", "video_prompt_3",
    "title", "description", "tags", "hashtags",
    "thumbnail_concept", "pinned_comment", "category", "language",
}

METADATA_TOOL = {
    "name": "submit_video_metadata",
    "description": "Submit the three scene prompts and all YouTube SEO metadata for the 30-second video.",
    "input_schema": {
        "type": "object",
        "properties": {
            "video_prompt_1": {
                "type": "string",
                "description": "Scene 1 (Hook, 0–10s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
            },
            "video_prompt_2": {
                "type": "string",
                "description": "Scene 2 (Amplification, 10–20s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
            },
            "video_prompt_3": {
                "type": "string",
                "description": "Scene 3 (Resolution, 20–30s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
            },
            "title": {
                "type": "string",
                "description": "YouTube title under 60 characters, keyword-first",
            },
            "description": {
                "type": "string",
                "description": "3-line structured description: hook, value, CTA with hashtags",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exactly 15 SEO-optimised YouTube tags",
            },
            "hashtags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exactly 5 hashtags without # symbol — 4 topic-specific plus Shorts",
            },
            "thumbnail_concept": {
                "type": "string",
                "description": "One sentence describing a high-contrast thumbnail for AI image generation",
            },
            "pinned_comment": {
                "type": "string",
                "description": "Engaging question or statement under 100 characters to pin as first comment",
            },
            "category": {
                "type": "string",
                "enum": ["Education", "Entertainment", "HowTo", "News", "Finance", "Health", "Technology", "Lifestyle"],
            },
            "language": {
                "type": "string",
                "description": "ISO 639-1 language code e.g. en",
            },
        },
        "required": [
            "video_prompt_1", "video_prompt_2", "video_prompt_3",
            "title", "description", "tags", "hashtags",
            "thumbnail_concept", "pinned_comment", "category", "language",
        ],
    },
}


def _build_user_prompt(trend: dict, niche: str, domain: str) -> str:
    return f"""Niche: {niche}
Domain focus: {domain}
Trend topic: {trend['topic']}
Hook (first sentence of video): {trend['hook']}
Target emotion: {trend['target_emotion']}
Key visual idea: {trend['key_visual_idea']}
Competitor angle: {trend['competitor_angle']}

Generate the three scene prompts and SEO-optimised YouTube metadata now."""


def _validate(data: dict) -> None:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    for i in (1, 2, 3):
        prompt = data.get(f"video_prompt_{i}", "")
        words = len(prompt.split())
        if words < 60:
            raise ValueError(f"video_prompt_{i} too short: {words} words (min 60)")
        prompt_lower = prompt.lower()
        for word in DISQUALIFIED_WORDS:
            if word in prompt_lower:
                raise ValueError(f"video_prompt_{i} contains disqualified word: {repr(word)}")

    if len(data.get("title", "")) > 60:
        raise ValueError(f"title too long: {len(data['title'])} chars (max 60)")

    if len(data.get("tags", [])) < 10:
        raise ValueError(f"Not enough tags: {len(data['tags'])} (min 10)")

    if len(data.get("hashtags", [])) < 3:
        raise ValueError(f"Not enough hashtags: {len(data['hashtags'])} (min 3)")

    if len(data.get("pinned_comment", "")) < 10:
        raise ValueError("pinned_comment too short")


def _enrich(data: dict) -> dict:
    hashtags: list = data.get("hashtags", [])
    if "Shorts" not in hashtags and "shorts" not in [h.lower() for h in hashtags]:
        hashtags.append("Shorts")
    data["hashtags"] = hashtags

    desc: str = data.get("description", "")
    hashtag_block = " ".join(f"#{h}" for h in hashtags)
    if "#Shorts" not in desc and "#shorts" not in desc.lower():
        data["description"] = desc.rstrip() + f"\n{hashtag_block}"

    data["category_id"] = CATEGORY_ID_MAP.get(data.get("category", "Finance"), "27")
    # Keep a combined video_prompt field for logging/db compatibility
    data["video_prompt"] = data["video_prompt_1"]
    return data


def _call_claude(client: anthropic.Anthropic, model: str, user_prompt: str) -> dict:
    message = client.messages.create(
        model=model,
        max_tokens=2400,
        system=SYSTEM_PROMPT,
        tools=[METADATA_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_prompt}],
    )
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_video_metadata":
            data = dict(block.input)
            logger.debug("Stage 3 | Tool input keys: %s", list(data.keys()))
            _validate(data)
            return _enrich(data)
    raise ValueError(f"Stage 3 | Claude did not call the metadata tool. Response: {message.content}")


def run(stage2_output: dict, dry_run: bool = False) -> dict:
    """Execute Stage 3. Returns 3 scene prompts and metadata dict."""
    conf = cfg.load_config()
    niche: str = conf["niche"]
    domain: str = conf["domain"]
    model: str = conf["claude_model"]

    logger.info("Stage 3 | Generating video prompts for topic: %s", stage2_output["topic"])

    if dry_run:
        logger.info("Stage 3 | DRY-RUN — returning mock prompts and metadata")
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
            result = _call_claude(client, model, user_prompt)
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
    scene1 = (
        "A young woman in a cosy coffee shop reaches the counter and taps her phone to pay. "
        "The camera opens in a tight close-up on the payment screen showing $6.50, then pulls "
        "back in a smooth tracking shot as she takes her oat-milk latte and walks to a window "
        "seat. Warm golden morning light floods through large glass panes, casting long amber "
        "shadows. The colour grade is rich and warm — honey tones with soft vignetting. Her "
        "expression is relaxed, completely unaware. The 9:16 vertical frame keeps her and the "
        "cup centred, intimate and personal."
    )
    scene2 = (
        "A kitchen table fills the frame, lit by cool, clinical overhead light. Hundreds of "
        "coffee cups are arranged in a neat grid covering the entire surface — a visual "
        "representation of 365 daily coffees. The camera slowly pushes in from a wide overhead "
        "shot, tightening on the centre of the grid. A hand enters frame and places one final "
        "cup. Crisp shadows fall across the cups. The colour grade shifts cooler — pale blue "
        "tones replacing warmth. The atmosphere is quiet and confrontational. The scale of the "
        "habit becomes undeniable."
    )
    scene3 = (
        "The same woman now stands at her own kitchen counter, pressing down a French press "
        "with a small satisfied smile. Steam rises from a ceramic mug beside a small jar of "
        "coins. The camera begins close on her hands and slowly rises to frame her face in warm "
        "natural window light — golden hour coming through sheer curtains. The colour grade "
        "returns to warm amber, matching scene one but feeling lighter, calmer, resolved. A "
        "small notebook on the counter shows hand-written numbers. The mood is quietly "
        "triumphant. She looks up and almost — almost — meets the camera."
    )
    return {
        "video_prompt_1": scene1,
        "video_prompt_2": scene2,
        "video_prompt_3": scene3,
        "video_prompt": scene1,
        "title": "Your $6 Coffee Is Costing $2,190 a Year",
        "description": (
            "Your daily coffee habit is quietly draining over $2,000 from your savings every year.\n"
            "Here's the exact visual math that will make you rethink your morning routine.\n"
            "Follow for daily money tips that actually move the needle. "
            "#PersonalFinance #SavingMoney #LatteFactor #MoneyHacks #Shorts"
        ),
        "tags": [
            "personal finance", "money tips", "financial advice",
            "how to save money fast", "money saving habits", "latte factor", "cut daily expenses",
            "why coffee is costing you money", "latte factor explained", "save $2000 a year", "daily spending audit",
            "personal finance for millennials", "money tips for beginners",
            "finance shorts", "money short video",
        ],
        "hashtags": ["PersonalFinance", "SavingMoney", "LatteFactor", "MoneyHacks", "Shorts"],
        "thumbnail_concept": (
            "Close-up of a shocked young woman's face (wide eyes, open mouth) on the left, "
            "with a bold red '$2,190' in large white text on a dark background on the right, "
            "a coffee cup icon beneath it. High contrast, warm orange accent colour."
        ),
        "pinned_comment": "How much do you spend on coffee a month? Drop the number — I'll calculate your yearly total ☕",
        "category": "Finance",
        "language": "en",
        "category_id": "27",
    }
