from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class TriggerDecision:
    intervene: bool
    active: bool
    cooldown_remaining: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdaptiveTrigger:
    """Stateful hysteresis + cooldown trigger for adaptive HITL governance."""

    threshold_high: float = 0.4
    threshold_low: float = 0.2
    cooldown_steps: int = 0
    active: bool = False
    cooldown_remaining: int = 0

    def evaluate(self, risk: float) -> TriggerDecision:
        risk = float(risk)
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if risk <= self.threshold_low:
                self.active = False
            return TriggerDecision(False, self.active, self.cooldown_remaining, "cooldown")

        if self.active:
            if risk <= self.threshold_low:
                self.active = False
                return TriggerDecision(False, False, self.cooldown_remaining, "risk_below_low_threshold")
            self.cooldown_remaining = int(max(0, self.cooldown_steps))
            return TriggerDecision(True, True, self.cooldown_remaining, "hysteresis_active")

        if risk >= self.threshold_high:
            self.active = True
            self.cooldown_remaining = int(max(0, self.cooldown_steps))
            return TriggerDecision(True, True, self.cooldown_remaining, "risk_above_high_threshold")

        return TriggerDecision(False, False, self.cooldown_remaining, "risk_below_high_threshold")
