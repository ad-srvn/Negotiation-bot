from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    issues: tuple[str, ...] = ("price", "quantity", "delivery_time", "service_level")
    n_steps_list: tuple[int, ...] = (12, 24, 36)
    profiles: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    strategies: tuple[str, ...] = (
        "BoulwareTBNegotiator",
        "LinearTBNegotiator",
        "ConcederTBNegotiator",
        "SmartAspirationNegotiator",
        "GeniusBOABaselineNegotiator",
    )
    n_seeds: int = 40
    output_dir: Path = Path("results/main")
    include_self_play: bool = True
    time_limit: float | None = None
    buyer_reserved_value: float = 0.35
    seller_reserved_value: float = 0.35
    pend: float = 0.04
    pend_per_second: float = 0.02
    bootstrap_resamples: int = 2000
    permutation_resamples: int = 2000
    progress_every: int = 100

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload
