"""Stage 3 — Video Prompt & Metadata Generator.

Takes the approved trend topic and generates five cinematic 10-second scene
prompts (forming a 50-second Short) plus SEO-optimised YouTube upload metadata.
"""

import os
import time

import anthropic

from utils import config as cfg
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are simultaneously a film director, a YouTube SEO specialist, and a viral short-form content strategist.

You will receive a trend topic and must produce two things in a single response:

PART 1 — FIVE SCENE PROMPTS FOR A 50-SECOND SHORT
Write five sequential cinematic prompts for a text-to-video AI model (Kling). Each scene is exactly 10 seconds. Together they form a complete 50-second story arc following this structure:

  Scene 1 (0–10s)  — HOOK: shocking number or counterintuitive claim
  Scene 2 (10–20s) — PROBLEM: why this affects the viewer specifically
  Scene 3 (20–30s) — SOLUTION part 1: introduce the actionable step
  Scene 4 (30–40s) — SOLUTION part 2 + PROOF: show the step in action with a real result
  Scene 5 (40–50s) — RESOLUTION: viewer feels empowered, grateful, ready to act

FACIAL EXPRESSION ARC — THIS IS MANDATORY IN EVERY VIDEO:
The subject's face must follow this exact emotional journey across the five scenes:
- Scene 1: shocked or surprised — a number just hit them
- Scene 2: visibly worried or concerned — they realise this affects them personally
- Scene 3: attentive and hopeful — they sense a way out
- Scene 4: nodding with dawning relief — the evidence confirms it works
- Scene 5: warm, genuine smile — they feel grateful and empowered
Every scene prompt MUST explicitly describe the subject's facial expression. A flat or neutral face is not acceptable.

SCENE 1 — HOOK (seconds 0–10)
Open mid-action with the shocking number or claim. Make the viewer feel something important is at stake immediately. Subject's face is shocked or surprised.

SCENE 2 — PROBLEM (seconds 10–20)
Show why this problem affects the viewer personally and specifically. Make the emotional weight land. Subject's face is visibly worried or concerned.

SCENE 3 — SOLUTION PART 1 (seconds 20–30)
Introduce the single clear, actionable step. Begin to show it being applied. Subject's face shifts to attentive and hopeful.

SCENE 4 — SOLUTION PART 2 + PROOF (seconds 30–40)
Complete the actionable step and show the concrete result — a number, a before/after, a real outcome. Subject's face shows nodding relief.

SCENE 5 — RESOLUTION (seconds 40–50)
End on a visual that implies positive change. The viewer feels empowered and ready to act. Subject's face shows a warm, genuine smile — the kind that says "I've got this now."

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
    "video_prompt_1", "video_prompt_2", "video_prompt_3", "video_prompt_4", "video_prompt_5",
    "title", "description", "tags", "hashtags",
    "thumbnail_concept", "pinned_comment", "category", "language",
}

METADATA_TOOL = {
    "name": "submit_video_metadata",
    "description": "Submit the five scene prompts and all YouTube SEO metadata for the 50-second video.",
    "input_schema": {
        "type": "object",
        "properties": {
            "video_prompt_1": {
                "type": "string",
                "description": "Scene 1 (Hook, 0–10s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
            },
            "video_prompt_2": {
                "type": "string",
                "description": "Scene 2 (Problem, 10–20s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
            },
            "video_prompt_3": {
                "type": "string",
                "description": "Scene 3 (Solution part 1, 20–30s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
            },
            "video_prompt_4": {
                "type": "string",
                "description": "Scene 4 (Solution part 2 + Proof, 30–40s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
            },
            "video_prompt_5": {
                "type": "string",
                "description": "Scene 5 (Resolution, 40–50s): cinematic prompt, 80-120 words, single paragraph, 9:16 vertical frame",
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
            "video_prompt_1", "video_prompt_2", "video_prompt_3", "video_prompt_4", "video_prompt_5",
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

    for i in (1, 2, 3, 4, 5):
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
    data["video_prompt"] = data["video_prompt_1"]  # logging/db compat
    return data


def _call_claude(client: anthropic.Anthropic, model: str, user_prompt: str) -> dict:
    message = client.messages.create(
        model=model,
        max_tokens=3500,
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
        "A young woman at a coffee shop counter taps her phone to pay — the screen flashes $6.50. "
        "The camera opens in a tight close-up on the receipt, then pulls back in a smooth tracking "
        "shot. Warm golden morning light floods through glass panes. Her expression is shocked, "
        "brows raised, lips slightly parted — a number just registered. The colour grade is rich "
        "honey tones with soft vignetting. The 9:16 vertical frame keeps her centred, intimate."
    )
    scene2 = (
        "A kitchen table fills the frame under cool clinical overhead light. Hundreds of coffee cups "
        "arranged in a grid — 365 daily coffees visualised. The camera slowly pushes in from a wide "
        "overhead shot. Her face appears reflected in a glass cup, expression visibly worried, brow "
        "furrowed. The colour grade shifts cooler, pale blue replacing warmth. The atmosphere is "
        "confrontational. The scale of the habit becomes undeniable and personal."
    )
    scene3 = (
        "The same woman stands at her kitchen counter, filling a French press. She pauses, pulls out "
        "her phone and opens a banking app. Her expression shifts to attentive and hopeful — eyes "
        "focused, leaning slightly forward. Warm natural window light, golden hour through sheer "
        "curtains. The colour grade warms back up. A notebook lies open beside the press. She taps "
        "something on the phone with quiet deliberation. The mood is purposeful."
    )
    scene4 = (
        "Close-up of a phone screen showing a high-yield savings account. A transfer of $100 is "
        "confirmed. The camera pulls back to reveal her face — nodding slowly, a dawning smile of "
        "relief spreading. A small jar of coins sits on the counter beside her. Studio softbox "
        "light, warm and clean. The colour grade is bright and optimistic. Numbers on a notepad "
        "beside her show a running total. The proof is visible and real."
    )
    scene5 = (
        "The woman sits at a sunlit kitchen table, mug of home-brewed coffee in hand, a genuine "
        "warm smile on her face — the kind that says I've got this now. A savings app on her phone "
        "shows a growing balance. The camera holds on her face in a slow gentle zoom. Golden hour "
        "light fills the frame. The colour grade is warm amber, soft and resolved. The mood is "
        "quietly triumphant. She looks up almost — almost — meeting the camera."
    )
    return {
        "video_prompt_1": scene1,
        "video_prompt_2": scene2,
        "video_prompt_3": scene3,
        "video_prompt_4": scene4,
        "video_prompt_5": scene5,
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
