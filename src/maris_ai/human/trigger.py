from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class TriggerDecision:
    intervene: bool
    active: bool
    cooldown_remaining: int
    reason: str
    suppressed: bool = False
    cooldown_released: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdaptiveTrigger:
    """Stateful hysteresis + cooldown trigger for adaptive HITL governance.

    v0.5 adds two safeguards requested during review:
    - hard critical risks bypass cooldown;
    - a continuously rising risk trend can release cooldown early.
    """

    threshold_high: float = 0.4
    threshold_low: float = 0.2
    cooldown_steps: int = 0
    trend_release_delta: float = 0.15
    active: bool = False
    cooldown_remaining: int = 0

    def evaluate(self, risk: float, *, critical: bool = False, risk_delta: float = 0.0) -> TriggerDecision:
        risk = float(risk)
        risk_delta = float(risk_delta)

        if critical:
            self.active = True
            self.cooldown_remaining = int(max(0, self.cooldown_steps))
            return TriggerDecision(True, True, self.cooldown_remaining, "critical_override", cooldown_released=True)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if risk_delta >= float(self.trend_release_delta) and risk >= self.threshold_high:
                self.active = True
                self.cooldown_remaining = int(max(0, self.cooldown_steps))
                return TriggerDecision(True, True, self.cooldown_remaining, "risk_trend_release", cooldown_released=True)
            if risk <= self.threshold_low:
                self.active = False
            return TriggerDecision(False, self.active, self.cooldown_remaining, "cooldown", suppressed=True)

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
