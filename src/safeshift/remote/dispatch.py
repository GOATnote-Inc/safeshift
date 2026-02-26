"""SSH dispatch to cloud GPUs — stub for v0.1."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SSHDispatcher:
    """Dispatch evaluation runs to remote GPU servers via SSH.

    Stub implementation for v0.1. Use API dispatch (VLLMExecutor with
    remote base_url) instead.
    """

    def __init__(self, host: str, user: str | None = None, key_path: str | None = None):
        self._host = host
        self._user = user
        self._key_path = key_path
        logger.info("SSHDispatcher initialized for %s (stub)", host)

    async def dispatch(self, command: str) -> str:
        raise NotImplementedError("SSH dispatch is a stub in v0.1")

    async def collect_results(self, remote_path: str, local_path: str) -> str:
        raise NotImplementedError("SSH result collection is a stub in v0.1")
