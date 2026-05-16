"""Stage 2b — Resource Finder.

Uses Claude to generate topic-specific resources for the pinned comment.
"""

import os

import anthropic

from utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "⚠️ This content is for entertainment purposes only and does not constitute "
    "financial, legal, or medical advice. Always consult a qualified professional."
)

RESOURCE_TOOL = {
    "name": "submit_resources",
    "description": "Submit a list of real, specific resources directly relevant to the video topic.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resources": {
                "type": "array",
                "description": "3–5 real resources that directly address the specific topic",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["title", "url"],
                },
                "minItems": 3,
                "maxItems": 5,
            }
        },
        "required": ["resources"],
    },
}


def run(stage2_out: dict, dry_run: bool = False) -> list[dict]:
    """Generate topic-specific resources via Claude. Returns list of {title, url} dicts."""
    topic: str = stage2_out.get("topic", "")

    if dry_run:
        logger.info("Stage 2b | DRY-RUN — skipping resource generation")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("Stage 2b | ANTHROPIC_API_KEY not set — skipping resources")
        return []

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            tools=[RESOURCE_TOOL],
            tool_choice={"type": "any"},
            messages=[{
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    "Provide 3–5 real, specific online resources that directly address this topic. "
                    "Prioritise official government pages (.gov), major nonprofits (aarp.org), "
                    "and well-known financial guidance sites. "
                    "Each URL must link to a specific page about this topic — not a generic homepage. "
                    "Only include resources you are confident actually exist."
                ),
            }],
        )
        for block in message.content:
            if block.type == "tool_use" and block.name == "submit_resources":
                resources = block.input.get("resources", [])
                if resources:
                    logger.info("Stage 2b | Generated %d topic-specific resources", len(resources))
                    return resources
        logger.warning("Stage 2b | Claude returned no resources")
        return []

    except Exception as exc:
        logger.warning("Stage 2b | Resource generation failed (%s) — skipping resources", exc)
        return []


def format_pinned_comment(resources: list[dict]) -> str:
    """Build the pinned comment: disclaimer → resource list → disclaimer."""
    lines = [DISCLAIMER, ""]
    if resources:
        lines.append("📚 Free resources on this topic:")
        for r in resources:
            lines.append(f"• {r['title']}")
            lines.append(f"  {r['url']}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
