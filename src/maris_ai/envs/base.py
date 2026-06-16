from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class AgentState:
    pos: np.ndarray
    vel: np.ndarray

@dataclass
class WorldState:
    agents: List[AgentState]
    t: int
    scenario: str

class BaseEnv:
    def __init__(self, dt: float = 0.2, arena_radius: float = 20.0):
        self.dt = float(dt)
        self.arena_radius = float(arena_radius)

    def reset(self, initial: WorldState) -> WorldState:
        return initial

    def step(self, s: WorldState, actions: List[np.ndarray]) -> WorldState:
        agents_next: List[AgentState] = []
        for ag, act in zip(s.agents, actions):
            pos = ag.pos + act * self.dt
            vel = act
            agents_next.append(AgentState(pos=pos, vel=vel))
        return WorldState(agents=agents_next, t=s.t + 1, scenario=s.scenario)
