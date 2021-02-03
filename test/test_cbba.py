#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import random
import math
import numpy as np
from dataclasses import dataclass
from CBBA import CBBA
from Agent import Agent
from Task import Task
from WorldInfo import WorldInfo
import HelperLibrary as hp


if __name__ == "__main__":
    # a json configuration file
    config_file_name = "config.json"

    # create a world
    WorldInfoTest = WorldInfo()
    WorldInfoTest.x_min = 0
    WorldInfoTest.x_max = 100
    WorldInfoTest.y_min = 0
    WorldInfoTest.y_max = 100
    WorldInfoTest.distance_max = math.sqrt((WorldInfoTest.x_max-WorldInfoTest.x_min)**2 + \
        (WorldInfoTest.y_max-WorldInfoTest.y_min)**2 + (WorldInfoTest.z_max-WorldInfoTest.z_min)**2)

    num_agents = 4
    num_tasks = 12

    AgentList, TaskList = hp.create_agents_and_tasks(num_agents, num_tasks, WorldInfoTest, config_file_name)



    # CBBA_solver = CBBA()
