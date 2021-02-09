#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import random
import math
import numpy as np
from dataclasses import dataclass, field
from CBBA import CBBA
from Agent import Agent
from Task import Task
from WorldInfo import WorldInfo
import HelperLibrary as hp


if __name__ == "__main__":
    # a json configuration file
    config_file_name = "config_example.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # create a world
    WorldInfoTest = WorldInfo([-2.0,2.5], [-1.5,5.5], [0.0,20.0])

    num_agents = 5
    num_tasks = 10

    # create a list of Agent(s) and Task(s)
    AgentList, TaskList = hp.create_agents_and_tasks(num_agents, num_tasks, WorldInfoTest, config_data)

    # create a CBBA solver
    CBBA_solver = CBBA(AgentList, TaskList, WorldInfoTest, config_data)

    # solve
    # score_total = CBBA_solver.solve()


    