#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
from dataclasses import dataclass


@dataclass
class Task:
    task_id: int = 0
    task_type: int = 0
    task_value: float = 0 # task reward
    start_time: float = 0 # task start time (sec)
    end_time: float = 0 # task expiry time (sec)
    duration: float = 0 # task default duration (sec)
    discount: float = 0.1 # task exponential discount (lambda)
    x: float = 0 # task position (meters)
    y: float = 0 # task position (meters)
    z: float = 0 # task position (meters)