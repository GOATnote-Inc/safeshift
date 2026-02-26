"""GPU inventory management — stub for v0.1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUNode:
    """A remote GPU node."""

    host: str
    gpu_type: str  # e.g., "A100-80GB", "H100"
    gpu_count: int
    available: bool = True
    vram_gb: float = 0.0


class GPUInventory:
    """Track available GPU nodes for remote dispatch.

    Stub implementation for v0.1.
    """

    def __init__(self):
        self._nodes: list[GPUNode] = []

    def add_node(self, node: GPUNode) -> None:
        self._nodes.append(node)

    def available_nodes(self) -> list[GPUNode]:
        return [n for n in self._nodes if n.available]

    def best_node_for(self, vram_required_gb: float) -> GPUNode | None:
        candidates = [n for n in self.available_nodes() if n.vram_gb >= vram_required_gb]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.vram_gb)
