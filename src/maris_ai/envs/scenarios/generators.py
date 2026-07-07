from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from maris_ai.envs.base import AgentState, WorldState

@dataclass(frozen=True)
class ScenarioParams:
    n_agents: int = 5
    arena_radius: float = 20.0

def _ring_positions(rng: np.random.Generator, n: int, r_max: float) -> List[np.ndarray]:
    out = []
    for _ in range(n):
        r = rng.uniform(0, r_max * 0.7)
        ang = rng.uniform(0, 2*np.pi)
        out.append(np.array([r*np.cos(ang), r*np.sin(ang)], dtype=float))
    return out

def head_on(seed: int, params: ScenarioParams) -> WorldState:
    rng = np.random.default_rng(seed)
    n = params.n_agents
    pos = _ring_positions(rng, n, params.arena_radius)
    if n >= 2:
        pos[0] = np.array([-8.0, 0.0])
        pos[1] = np.array([ 8.0, 0.0])
    agents = [AgentState(pos=pos[i], vel=rng.normal(0, 0.2, size=(2,)).astype(float)) for i in range(n)]
    return WorldState(agents=agents, t=0, scenario="head_on")

def crossing(seed: int, params: ScenarioParams) -> WorldState:
    rng = np.random.default_rng(seed)
    n = params.n_agents
    pos = _ring_positions(rng, n, params.arena_radius)
    if n >= 2:
        pos[0] = np.array([-7.0, -2.0])
        pos[1] = np.array([ 0.0,  7.0])
    agents = [AgentState(pos=pos[i], vel=rng.normal(0, 0.2, size=(2,)).astype(float)) for i in range(n)]
    return WorldState(agents=agents, t=0, scenario="crossing")

def overtaking(seed: int, params: ScenarioParams) -> WorldState:
    rng = np.random.default_rng(seed)
    n = params.n_agents
    pos = _ring_positions(rng, n, params.arena_radius)
    if n >= 2:
        pos[0] = np.array([-8.0, -1.0])
        pos[1] = np.array([-4.0, -1.0])
    agents = [AgentState(pos=pos[i], vel=rng.normal(0, 0.2, size=(2,)).astype(float)) for i in range(n)]
    return WorldState(agents=agents, t=0, scenario="overtaking")

def bottleneck(seed: int, params: ScenarioParams) -> WorldState:
    rng = np.random.default_rng(seed)
    n = params.n_agents
    agents = []
    for _ in range(n):
        pos = rng.normal(0, 2.0, size=(2,)).astype(float)
        vel = rng.normal(0, 0.2, size=(2,)).astype(float)
        agents.append(AgentState(pos=pos, vel=vel))
    return WorldState(agents=agents, t=0, scenario="bottleneck")

def restricted_zone(seed: int, params: ScenarioParams) -> WorldState:
    rng = np.random.default_rng(seed)
    n = params.n_agents
    pos = _ring_positions(rng, n, params.arena_radius)
    agents = [AgentState(pos=pos[i], vel=rng.normal(0, 0.2, size=(2,)).astype(float)) for i in range(n)]
    return WorldState(agents=agents, t=0, scenario="restricted_zone")


def high_density(seed: int, params: ScenarioParams) -> WorldState:
    """Stress-test scenario: many vessels initialized in a compact traffic region."""
    rng = np.random.default_rng(seed)
    n = params.n_agents
    agents = []
    for i in range(n):
        angle = 2 * np.pi * i / max(1, n)
        radius = rng.uniform(1.0, min(4.0, params.arena_radius * 0.25))
        pos = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float) + rng.normal(0, 0.25, size=(2,))
        # inward-biased velocities create competing close-quarters decisions
        vel = -0.2 * pos / (np.linalg.norm(pos) + 1e-12) + rng.normal(0, 0.15, size=(2,))
        agents.append(AgentState(pos=pos.astype(float), vel=vel.astype(float)))
    return WorldState(agents=agents, t=0, scenario="high_density")


def sudden_emergency(seed: int, params: ScenarioParams) -> WorldState:
    """Stress-test proxy for abrupt close-quarters risk escalation.

    The base simulator has no dynamic obstacle object, so this scenario starts
    several agents on converging trajectories near the same decision region.
    """
    rng = np.random.default_rng(seed)
    n = params.n_agents
    agents = []
    anchors = [np.array([-2.5, 0.0]), np.array([2.5, 0.0]), np.array([0.0, -2.5]), np.array([0.0, 2.5])]
    for i in range(n):
        base = anchors[i % len(anchors)] + rng.normal(0, 0.25, size=(2,))
        vel = -0.35 * base / (np.linalg.norm(base) + 1e-12) + rng.normal(0, 0.12, size=(2,))
        agents.append(AgentState(pos=base.astype(float), vel=vel.astype(float)))
    return WorldState(agents=agents, t=0, scenario="sudden_emergency")


def boundary_stress(seed: int, params: ScenarioParams) -> WorldState:
    """Stress-test scenario with agents near the operational boundary."""
    rng = np.random.default_rng(seed)
    n = params.n_agents
    agents = []
    radius = params.arena_radius * 0.85
    for i in range(n):
        angle = 2 * np.pi * i / max(1, n)
        pos = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float) + rng.normal(0, 0.4, size=(2,))
        # outward component pressures boundary governance
        vel = 0.35 * pos / (np.linalg.norm(pos) + 1e-12) + rng.normal(0, 0.10, size=(2,))
        agents.append(AgentState(pos=pos.astype(float), vel=vel.astype(float)))
    return WorldState(agents=agents, t=0, scenario="boundary_stress")

SCENARIOS = {
    "head_on": head_on,
    "crossing": crossing,
    "overtaking": overtaking,
    "bottleneck": bottleneck,
    "restricted_zone": restricted_zone,
    "high_density": high_density,
    "sudden_emergency": sudden_emergency,
    "boundary_stress": boundary_stress,
}
