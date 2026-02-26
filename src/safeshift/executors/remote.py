"""Remote dispatch wrapper — stub for SSH/API dispatch."""

from __future__ import annotations

import logging

from safeshift.executor import Executor, ExecutorResult

logger = logging.getLogger(__name__)


class RemoteExecutor(Executor):
    """SSH/API dispatch to remote GPU servers.

    Stub implementation — will be fully implemented in v0.2.
    At v0.1, use the API executor with a remote vLLM endpoint instead.
    """

    def __init__(self, remote_url: str, **kwargs):
        self._remote_url = remote_url
        logger.info("RemoteExecutor initialized for %s (stub)", remote_url)

    @property
    def name(self) -> str:
        return "remote"

    async def execute(
        self,
        messages: list[dict[str, str]],
        model: str,
        optimization: str = "baseline",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 4096,
    ) -> ExecutorResult:
        raise NotImplementedError(
            "RemoteExecutor is a stub in v0.1. "
            "Use VLLMExecutor with a remote base_url instead: "
            "safeshift run --executor vllm --remote http://host:8000/v1"
        )
