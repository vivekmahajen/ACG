"""Stage 2b — Resource Finder.

Uses Claude with web search to find real, topic-specific articles and resources.
Falls back to a curated whitelist of trusted domains if web search is unavailable.
"""

import os

import anthropic

from utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "⚠️ This content is for entertainment purposes only and does not constitute "
    "financial, legal, or medical advice. Always consult a qualified professional."
)

# Fallback whitelist — used only if web search fails/is unavailable
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
    "investopedia.com",
    "nerdwallet.com",
    "kiplinger.com",
    "moneywise.com",
    "consumerreports.org",
]

RESOURCE_TOOL = {
    "name": "submit_resources",
    "description": "Submit the final list of resources found for this topic.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resources": {
                "type": "array",
                "description": "3-5 credible, topic-specific resources with real URLs",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Descriptive title, e.g. 'Investopedia — How Loyalty Programs Work Against Consumers'",
                        },
                        "url": {
                            "type": "string",
                            "description": "The full URL of the article or page",
                        },
                    },
                    "required": ["title", "url"],
                },
                "minItems": 1,
                "maxItems": 5,
            }
        },
        "required": ["resources"],
    },
}

_FALLBACK_DOMAIN_LIST = "\n".join(f"- https://www.{d}" for d in TRUSTED_DOMAINS)


def run(stage2_out: dict, dry_run: bool = False) -> list[dict]:
    """Generate topic-specific resources. Returns list of {title, url} dicts."""
    topic: str = stage2_out.get("topic", "")

    if dry_run:
        logger.info("Stage 2b | DRY-RUN — skipping resource generation")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("Stage 2b | ANTHROPIC_API_KEY not set — skipping resources")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    # Try web search first for real article URLs
    resources = _search_resources(client, topic)
    if resources:
        return resources

    # Fall back to whitelist-constrained generation
    logger.info("Stage 2b | Falling back to whitelist-constrained resources")
    return _whitelist_resources(client, topic)


def _search_resources(client, topic: str) -> list[dict]:
    """Use Claude with web search to find real article URLs for the topic."""
    prompt = (
        f"Search the web to find 3 to 5 specific, credible articles or web pages about this topic:\n"
        f"{topic}\n\n"
        "Look for articles from reputable sources such as consumer finance publications, "
        "major news outlets, government health sites, university research, or well-known "
        "personal finance websites. Prioritize articles that are directly about the topic — "
        "not just tangentially related.\n\n"
        "After searching, use the submit_resources tool to provide the exact URLs and "
        "descriptive titles of what you found. Only include resources that genuinely cover this topic."
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        for turn in range(8):
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                tools=[
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
                    RESOURCE_TOOL,
                ],
                messages=messages,
            )
            logger.info("Stage 2b | Web search turn %d: stop_reason=%s", turn + 1, message.stop_reason)

            # Look for our structured output in this turn
            for block in message.content:
                if block.type == "tool_use" and block.name == "submit_resources":
                    resources = block.input.get("resources", [])
                    if resources:
                        logger.info("Stage 2b | Web search found %d resources", len(resources))
                        return resources

            if message.stop_reason == "end_turn":
                break

            # Continue the conversation — append assistant response and handle tool results
            messages.append({"role": "assistant", "content": message.content})

            tool_results = []
            for block in message.content:
                if block.type == "tool_use" and block.name != "submit_resources":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Search complete. Now use the submit_resources tool to provide the resources you found.",
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            elif message.stop_reason == "tool_use":
                break

        logger.warning("Stage 2b | Web search completed without submit_resources call")
        return []

    except anthropic.BadRequestError as e:
        logger.warning("Stage 2b | Web search not available: %s", e)
        return []
    except Exception as e:
        logger.warning("Stage 2b | Web search failed: %s", e)
        return []


def _whitelist_resources(client, topic: str) -> list[dict]:
    """Fallback: constrain Claude to root URLs from the trusted domain whitelist."""
    prompts = [
        (
            f"Topic: {topic}\n\n"
            "Provide 3–5 resources from this approved domain list that are DIRECTLY relevant to this specific topic.\n"
            f"{_FALLBACK_DOMAIN_LIST}\n\n"
            "IMPORTANT rules:\n"
            "- Only include a domain if it genuinely covers this topic.\n"
            "- Use only root or stable section URLs. Do NOT invent deep links.\n"
            "- Write a title that describes what the viewer will find there for THIS topic."
        ),
        (
            f"Topic: {topic}\n\n"
            "Pick 1–3 domains from this list most directly related to this topic, and provide their root URL.\n"
            f"{_FALLBACK_DOMAIN_LIST}\n\n"
            "Fewer relevant resources is better than more irrelevant ones."
        ),
        (
            f"Topic: {topic}\n\n"
            "Which single domain from this list is most relevant to this topic? Provide its root URL.\n"
            f"{_FALLBACK_DOMAIN_LIST}"
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
                        logger.info("Stage 2b | Whitelist fallback got %d resources (attempt %d)", len(resources), attempt)
                        return resources
            logger.warning("Stage 2b | Whitelist attempt %d returned no valid resources", attempt)
        except Exception as exc:
            logger.warning("Stage 2b | Whitelist attempt %d failed: %s", attempt, exc)

    logger.error("Stage 2b | All resource attempts failed for topic: %s", topic)
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
