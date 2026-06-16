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

SCENARIOS = {"head_on": head_on, "crossing": crossing, "overtaking": overtaking, "bottleneck": bottleneck, "restricted_zone": restricted_zone}
