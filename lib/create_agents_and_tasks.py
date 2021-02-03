#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import random
import numpy as np
from dataclasses import dataclass
from CBBA import CBBA
from Agent import Agent
from Task import Task


def create_agents_and_tasks(num_agents: int, num_tasks: int, world_info, config_file_name: str):
    """
    Generate agents and tasks based on a json configuration file.
    """

    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # Create some default agents

    # quad
    agent_quad_default = Agent()
    # agent type
    agent_quad_default.agent_type = config_data["AGENT_TYPES"].index("quad")
    # agent cruise velocity (m/s)
    agent_quad_default.nom_velocity = float(config_data["QUAD_DEFAULT"]["NOM_VELOCITY"])
    # agent fuel penalty (per meter)
    agent_quad_default.fuel = float(config_data["QUAD_DEFAULT"]["FUEL"])

    # car
    agent_car_default = Agent()
    # agent type
    agent_car_default.agent_type = config_data["AGENT_TYPES"].index("car")
    # agent cruise velocity (m/s)
    agent_car_default.nom_velocity = float(config_data["CAR_DEFAULT"]["NOM_VELOCITY"])
    # agent fuel penalty (per meter)
    agent_car_default.fuel = float(config_data["CAR_DEFAULT"]["FUEL"])


    # Create some default tasks

    # Track
    task_track_default = Task()
    # task type
    task_track_default.task_type = config_data["TASK_TYPES"].index("track")
    # task reward
    task_track_default.task_value = float(config_data["TRACK_DEFAULT"]["TASK_VALUE"])
    # task start time (sec)
    task_track_default.start_time = float(config_data["TRACK_DEFAULT"]["START_TIME"])
    # task expiry time (sec)
    task_track_default.end_time = float(config_data["TRACK_DEFAULT"]["START_TIME"])
    # task default duration (sec)
    task_track_default.duration = float(config_data["TRACK_DEFAULT"]["START_TIME"])

    # Rescue
    task_track_default = Task()
    # task type
    task_track_default.task_type = config_data["TASK_TYPES"].index("rescue")
    # task reward
    task_track_default.task_value = float(config_data["RESCUE_DEFAULT"]["TASK_VALUE"])
    # task start time (sec)
    task_track_default.start_time = float(config_data["RESCUE_DEFAULT"]["START_TIME"])
    # task expiry time (sec)
    task_track_default.end_time = float(config_data["RESCUE_DEFAULT"]["START_TIME"])
    # task default duration (sec)
    task_track_default.duration = float(config_data["RESCUE_DEFAULT"]["START_TIME"])

    # create empty list, each element is a dataclass Agent() or Task()
    AgentList = []
    TaskList = []

    # here is different than original CBBA codes

    # create random agents (quad only)
    for idx_agent in range(0, num_agents):
        AgentList.append(task_track_default)
        AgentList[idx_agent].agent_id = idx_agent
        AgentList[idx_agent].x = random.randint(world_info.x_min, world_info.x_max)
        AgentList[idx_agent].y = random.randint(world_info.y_min, world_info.y_max)
        AgentList[idx_agent].z = 0

    # create random tasks (track only)
    for idx_task in range(0, num_tasks):
        TaskList.append(agent_quad_default)
        TaskList[idx_task].task_id = idx_task
        TaskList[idx_task].x = random.randint(world_info.x_min, world_info.x_max)
        TaskList[idx_task].y = random.randint(world_info.y_min, world_info.y_max)
        TaskList[idx_task].z = 0

    return AgentList, TaskList