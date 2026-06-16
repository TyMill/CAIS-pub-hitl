from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkloadState:
    """Minimal deterministic operator workload accumulator."""

    interventions: int = 0
    delay_sum: float = 0.0

    def update(self, intervened: bool, delay_steps: int) -> None:
        if intervened:
            self.interventions += 1
            self.delay_sum += float(delay_steps)

    def workload(self, steps: int) -> float:
        return float(self.interventions / max(1, steps))

    def mean_delay(self) -> float:
        return float(self.delay_sum / max(1, self.interventions))
