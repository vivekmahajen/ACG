"""Stage 2b — Resource Finder.

Searches the web for authoritative resources related to today's video topic
so viewers can act on the advice. Results are included in the pinned comment.
"""

import re

from utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "⚠️ This content is for entertainment purposes only and does not constitute "
    "financial, legal, or medical advice. Always consult a qualified professional."
)

MAX_RESOURCES = 5

TRUSTED_DOMAINS = {
    "ssa.gov", "medicare.gov", "medicaid.gov", "usa.gov", "benefits.gov",
    "aarp.org", "nolo.com", "investopedia.com", "fool.com", "kiplinger.com",
    "nerdwallet.com", "bankrate.com", "consumerfinance.gov", "ftc.gov",
    "sec.gov", "irs.gov", "dol.gov", "hhs.gov", "healthline.com",
    "mayoclinic.org", "nih.gov", "cdc.gov",
}


def run(stage2_out: dict, dry_run: bool = False) -> list[dict]:
    """Search the web for resources on the topic. Returns list of {title, url} dicts."""
    topic: str = stage2_out.get("topic", "")

    if dry_run:
        logger.info("Stage 2b | DRY-RUN — returning mock resources")
        return _mock_resources()

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("Stage 2b | duckduckgo-search not installed — using mock resources")
        return _mock_resources()

    query = f"{topic} how to official resources guide"
    logger.info("Stage 2b | Searching: %r", query)

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=20))

        candidates: list[dict] = []
        for r in raw:
            url = r.get("href") or r.get("url", "")
            title = (r.get("title") or "").strip()
            if not url or not title:
                continue
            domain = _extract_domain(url)
            candidates.append({
                "title": title,
                "url": url,
                "_trusted": domain in TRUSTED_DOMAINS,
            })

        # Trusted domains first, then rest; cap at MAX_RESOURCES
        candidates.sort(key=lambda x: (not x["_trusted"], x["title"]))
        resources = [{"title": c["title"], "url": c["url"]} for c in candidates[:MAX_RESOURCES]]
        logger.info("Stage 2b | Found %d resources", len(resources))
        return resources if resources else _mock_resources()

    except Exception as exc:
        logger.warning("Stage 2b | Search failed (%s) — using mock resources", exc)
        return _mock_resources()


def format_pinned_comment(resources: list[dict]) -> str:
    """Build the pinned comment: disclaimer → resource list → disclaimer."""
    lines = [DISCLAIMER, ""]
    lines.append("📚 Learn more from these resources:")
    for r in resources:
        lines.append(f"• {r['title']}\n  {r['url']}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def _extract_domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1).lower() if m else ""


def _mock_resources() -> list[dict]:
    return [
        {"title": "Social Security Administration — Retirement Benefits", "url": "https://www.ssa.gov/benefits/retirement/"},
        {"title": "Medicare.gov — Official Medicare Site", "url": "https://www.medicare.gov/"},
        {"title": "AARP — Financial Guidance for Older Adults", "url": "https://www.aarp.org/money/"},
        {"title": "Consumer Financial Protection Bureau", "url": "https://www.consumerfinance.gov/"},
        {"title": "USA.gov — Benefits for Older Adults", "url": "https://www.usa.gov/benefits/"},
    ]
