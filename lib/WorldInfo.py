#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
from dataclasses import dataclass, field


@dataclass
class WorldInfo:
    x_min: int = 0 # min x coordinate [meter]
    x_max: int = 0 # max x coordinate [meter]
    y_min: int = 0 # min y coordinate [meter]
    y_max: int = 0 # max y coordinate [meter]
    z_min: int = 0 # min z coordinate [meter]
    z_max: int = 0 # max z coordinate [meter]
    distance_max: float = 0
