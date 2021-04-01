#!/usr/bin/env python3
from dataclasses import dataclass, field
import math


@dataclass
class WorldInfo:
    limit_x: list = field(default_factory=lambda: [0, 100])  # [min, max] x coordinate [meter]
    limit_y: list = field(default_factory=lambda: [0, 100])  # [min, max] y coordinate [meter]
    limit_z: list = field(default_factory=lambda: [0, 0])  # [min, max] z coordinate [meter]
    distance_max: float = 0

    def __init__(self, limit_x: list, limit_y: list, limit_z: list):
        self.limit_x = limit_x
        self.limit_y = limit_y
        self.limit_z = limit_z
        self.distance_max = math.sqrt((self.limit_x[1]-self.limit_x[0])**2 + (self.limit_y[1]-self.limit_y[0])**2 +
                                      (self.limit_z[1]-self.limit_z[0])**2)
