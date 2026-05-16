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
                "description": "3-5 resources relevant to the topic, using only approved trusted domains",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Descriptive title explaining what to find there, e.g. 'SSA.gov — Medicare cost estimator'",
                        },
                        "url": {
                            "type": "string",
                            "description": (
                                "URL from an approved domain. Use ONLY the root domain or one known stable "
                                "section (e.g. https://www.ssa.gov or https://www.ssa.gov/benefits/). "
                                "Do NOT invent deep links or specific page paths — they may not exist."
                            ),
                        },
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

_DOMAIN_LIST = "\n".join(f"- https://www.{d}" for d in TRUSTED_DOMAINS)


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

    client = anthropic.Anthropic(api_key=api_key)

    prompts = [
        # Attempt 1: strict relevance + root URLs only
        (
            f"Topic: {topic}\n\n"
            "Provide 3–5 resources from this approved domain list that are DIRECTLY relevant to this specific topic.\n"
            f"{_DOMAIN_LIST}\n\n"
            "IMPORTANT rules:\n"
            "- Only include a domain if it genuinely covers this topic. Do NOT include ssa.gov, medicare.gov, or any other "
            "domain just to fill the list — only include it if the topic is specifically about that subject.\n"
            "- Use only root or stable section URLs. Do NOT invent deep links.\n"
            "- Write a title that describes what the viewer will find there for THIS specific topic."
        ),
        # Attempt 2: even stricter on relevance
        (
            f"Topic: {topic}\n\n"
            "Pick 1–3 domains from this list that are most directly related to this topic, and provide their root URL.\n"
            f"{_DOMAIN_LIST}\n\n"
            "Only include domains genuinely relevant to the topic. Fewer relevant resources is better than more irrelevant ones."
        ),
        # Attempt 3: minimal — just 1 resource
        (
            f"Topic: {topic}\n\n"
            "Which single domain from this list is most relevant to this topic? Provide its root URL.\n"
            f"{_DOMAIN_LIST}"
        ),
    ]

    for attempt, prompt in enumerate(prompts, start=1):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                tools=[RESOURCE_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in message.content:
                if block.type == "tool_use" and block.name == "submit_resources":
                    resources = block.input.get("resources", [])
                    resources = [r for r in resources if _is_trusted(r.get("url", ""))]
                    if resources:
                        logger.info("Stage 2b | Got %d topic-specific resources (attempt %d)", len(resources), attempt)
                        return resources
            logger.warning("Stage 2b | Attempt %d returned no valid resources", attempt)
        except Exception as exc:
            logger.warning("Stage 2b | Attempt %d failed: %s", attempt, exc)

    logger.error("Stage 2b | All attempts failed — no resources for topic: %s", topic)
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
