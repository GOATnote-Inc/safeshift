"""OpenEM context enrichment — optional dependency."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def enrich_scenario_with_openem(
    scenario_messages: list[dict[str, str]],
    condition: str,
    max_chars: int = 2000,
) -> list[dict[str, str]]:
    """Enrich scenario messages with OpenEM clinical context.

    Optional dependency — returns messages unchanged if openem is not installed.
    """
    try:
        from openem.bridge import OpenEMBridge
    except ImportError:
        logger.debug("openem not installed, skipping enrichment")
        return scenario_messages

    try:
        bridge = OpenEMBridge()
        context = bridge.get_context(condition, max_chars=max_chars)
        if not context:
            return scenario_messages

        # Prepend context as system message or append to existing system
        enriched = list(scenario_messages)
        context_block = f"\n\n[Clinical Reference — {condition}]\n{context}\n"

        if enriched and enriched[0]["role"] == "system":
            enriched[0] = {
                "role": "system",
                "content": enriched[0]["content"] + context_block,
            }
        else:
            enriched.insert(0, {"role": "system", "content": context_block.strip()})

        return enriched

    except Exception as e:
        logger.warning("OpenEM enrichment failed for %s: %s", condition, e)
        return scenario_messages
