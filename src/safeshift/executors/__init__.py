"""Executor implementations."""

from safeshift.executors.api import APIExecutor
from safeshift.executors.mock import MockExecutor
from safeshift.executors.vllm import VLLMExecutor

EXECUTORS = {
    "mock": MockExecutor,
    "api": APIExecutor,
    "vllm": VLLMExecutor,
}


def get_executor(name: str, **kwargs):
    """Get an executor by name."""
    if name not in EXECUTORS:
        raise ValueError(f"Unknown executor: {name}. Available: {list(EXECUTORS.keys())}")
    return EXECUTORS[name](**kwargs)
