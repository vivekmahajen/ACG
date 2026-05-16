"""Stage 2b — Resource Finder.

Uses Claude to generate topic-specific resources from a curated whitelist of
domains that are known to be accessible from cloud servers. No live URL
validation — instead we constrain Claude to only use root URLs of trusted,
stable sites and format them as search links so they always resolve.
"""

import os

import anthropic

from utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "⚠️ This content is for entertainment purposes only and does not constitute "
    "financial, legal, or medical advice. Always consult a qualified professional."
)

# Domains that block server-side requests (Cloudflare, aggressive bot protection)
# are excluded. These domains are stable, accessible, and relevant to the niche.
TRUSTED_DOMAINS = [
    "ssa.gov",
    "medicare.gov",
    "medicaid.gov",
    "benefits.gov",
    "usa.gov",
    "consumerfinance.gov",
    "ftc.gov",
    "irs.gov",
    "dol.gov",
    "hhs.gov",
    "nih.gov",
    "cdc.gov",
    "aarp.org",
    "ncoa.org",
    "eldercare.acl.gov",
]

RESOURCE_TOOL = {
    "name": "submit_resources",
    "description": "Submit topic-specific resources using only the approved trusted domains.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resources": {
                "type": "array",
                "description": "5 resources relevant to the topic, using only approved trusted domains",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short descriptive title for this resource"},
                        "url": {"type": "string", "description": "Full URL — must be from one of the approved domains"},
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

_DOMAIN_LIST = "\n".join(f"- {d}" for d in TRUSTED_DOMAINS)


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
            max_tokens=600,
            tools=[RESOURCE_TOOL],
            tool_choice={"type": "any"},
            messages=[{
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    "Provide 3–5 resources directly relevant to this topic. "
                    "You MUST only use URLs from these approved domains:\n"
                    f"{_DOMAIN_LIST}\n\n"
                    "Link to specific pages about the topic where possible. "
                    "If no specific page exists, use the most relevant section of the site. "
                    "Do not use any domain not on this list."
                ),
            }],
        )
        for block in message.content:
            if block.type == "tool_use" and block.name == "submit_resources":
                resources = block.input.get("resources", [])
                # Filter to confirmed trusted domains only
                resources = [r for r in resources if _is_trusted(r.get("url", ""))]
                if resources:
                    logger.info("Stage 2b | Generated %d topic-specific resources", len(resources))
                    return resources

        logger.warning("Stage 2b | Claude returned no resources")
        return []

    except Exception as exc:
        logger.warning("Stage 2b | Resource generation failed (%s) — skipping resources", exc)
        return []


def _is_trusted(url: str) -> bool:
    import re
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    if not m:
        return False
    domain = m.group(1).lower()
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_DOMAINS)


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
