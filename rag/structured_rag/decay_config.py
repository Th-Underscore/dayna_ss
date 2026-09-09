from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecayConfig:
    method: str = "path_min_gated"
    decay_exponent: float = 0.4
    broad_threshold: int = 42
    final_threshold: int = 75

    convergence_enabled: bool = True
    pull_rate: float = 0.4

    re_traverse_enabled: bool = True
    re_traverse_threshold: int = 65
    re_traverse_min_importance: int = 42

    max_depth: int = 10
