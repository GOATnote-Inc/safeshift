"""OpenEM context enrichment — optional dependency."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _default_index_dir() -> Optional[Path]:
    """Discover the OpenEM index directory from the installed package location."""
    try:
        import openem

        pkg_dir = Path(openem.__file__).resolve().parent
        # openem package lives at <repo>/python/openem/
        # index lives at <repo>/data/index/
        repo_root = pkg_dir.parent.parent
        index_dir = repo_root / "data" / "index"
        if (index_dir / "manifest.json").exists():
            return index_dir
    except (ImportError, AttributeError):
        pass
    return None


def enrich_scenario_with_openem(
    scenario_messages: list[dict[str, str]],
    condition: str,
    max_chars: int = 2000,
    index_dir: Optional[str | Path] = None,
) -> list[dict[str, str]]:
    """Enrich scenario messages with OpenEM clinical context.

    Optional dependency — returns messages unchanged if openem is not installed.

    Args:
        scenario_messages: List of message dicts with "role" and "content".
        condition: Condition name to retrieve context for.
        max_chars: Max characters of clinical context to inject.
        index_dir: Path to OpenEM index directory. Auto-discovered if not provided.
    """
    try:
        from openem import OpenEMIndex
        from openem.bridge import OpenEMBridge
    except ImportError:
        logger.debug("openem not installed, skipping enrichment")
        return scenario_messages

    try:
        resolved_dir = Path(index_dir) if index_dir else _default_index_dir()
        if resolved_dir is None:
            logger.debug("OpenEM index directory not found, skipping enrichment")
            return scenario_messages

        idx = OpenEMIndex(resolved_dir)
        bridge = OpenEMBridge(idx)
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
