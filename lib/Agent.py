#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
from dataclasses import dataclass, field


@dataclass
class Agent:
    agent_id: int = 0
    agent_type: int = 0
    availability: float = 0 # agent availability (expected time in sec)
    # clr_plotting: list = field([0, 0, 0]) # for plotting
    x: float = 0 # task position (meters)
    y: float = 0 # task position (meters)
    z: float = 0 # task position (meters)
    nom_velocity: float = 0 # agent cruise velocity (m/s)
    fuel: float = 0 # agent fuel penalty (per meter)
