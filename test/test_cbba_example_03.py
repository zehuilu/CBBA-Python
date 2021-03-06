#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from CBBA import CBBA
from Agent import Agent
from Task import Task
from WorldInfo import WorldInfo
import HelperLibrary as hp


if __name__ == "__main__":
    # a json configuration file
    config_file_name = "config_example_03.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # create a world, each list is [min, max] coordinates for x,y,z axis
    WorldInfoTest = WorldInfo([-2.0,2.5], [-1.5,5.5], [0.0,20.0])

    # create a list of Agent(s) and Task(s)
    num_agents = 5
    num_tasks = 20
    max_depth = num_tasks
    AgentList, TaskList = hp.create_agents_and_tasks_homogeneous(num_agents, num_tasks, WorldInfoTest, config_data)

    # create a CBBA solver
    CBBA_solver = CBBA(config_data)

    t_start = time.time()

    # solve
    path_list, times_list = CBBA_solver.solve(AgentList, TaskList, WorldInfoTest, max_depth, time_window_flag=True)
    
    t_end = time.time()
    t_used = t_end - t_start
    print("Time used [sec]: ", t_used)


    # the output is CBBA_solver.path_list or path_list
    print("bundle_list")
    print(CBBA_solver.bundle_list)
    print("path_list")
    print(path_list)
    print("times_list")
    print(times_list)
    print("winners_list")
    print(CBBA_solver.winners_list)

    print("bid_list")
    print(CBBA_solver.bid_list)
    print("winner_bid_list")
    print(CBBA_solver.winner_bid_list)


    # plot without time window
    CBBA_solver.plot_assignment_without_timewindow()
    plt.show()
    