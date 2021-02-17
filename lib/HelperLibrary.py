#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import math
import random
import numpy as np
from dataclasses import dataclass, field
from Agent import Agent
from Task import Task
from WorldInfo import WorldInfo


def create_agents_and_tasks(num_agents: int, num_tasks: int, WorldInfoInput: WorldInfo, config_data):
    """
    Generate agents and tasks based on a json configuration file.

    config_data:
        config_file_name = "config.json"
        json_file = open(config_file_name)
        config_data = json.load(json_file)

    """

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
    task_track_default.end_time = float(config_data["TRACK_DEFAULT"]["END_TIME"])
    # task default duration (sec)
    task_track_default.duration = float(config_data["TRACK_DEFAULT"]["DURATION"])

    # Rescue
    task_rescue_default = Task()
    # task type
    task_rescue_default.task_type = config_data["TASK_TYPES"].index("rescue")
    # task reward
    task_rescue_default.task_value = float(config_data["RESCUE_DEFAULT"]["TASK_VALUE"])
    # task start time (sec)
    task_rescue_default.start_time = float(config_data["RESCUE_DEFAULT"]["START_TIME"])
    # task expiry time (sec)
    task_rescue_default.end_time = float(config_data["RESCUE_DEFAULT"]["END_TIME"])
    # task default duration (sec)
    task_rescue_default.duration = float(config_data["RESCUE_DEFAULT"]["DURATION"])

    # create empty list, each element is a dataclass Agent() or Task()
    AgentList = []
    TaskList = []


    # original CBBA codes
    # create random agents
    for idx_agent in range(0, num_agents):
        # create a new instance of dataclass agent_quad_default
        
        if (idx_agent/num_agents <= 0.5):
            AgentList.append(Agent(**agent_quad_default.__dict__))
        else:
            AgentList.append(Agent(**agent_car_default.__dict__))

        # AgentList.append(Agent(**agent_quad_default.__dict__))
        AgentList[idx_agent].agent_id = idx_agent
        AgentList[idx_agent].x = random.uniform(WorldInfoInput.limit_x[0], WorldInfoInput.limit_x[1])
        AgentList[idx_agent].y = random.uniform(WorldInfoInput.limit_y[0], WorldInfoInput.limit_y[1])
        AgentList[idx_agent].z = 0

    # create random tasks (track only)
    for idx_task in range(0, num_tasks):
        # create a new instance of dataclass task_track_default
        if (idx_task/num_tasks <= 0.5):
            TaskList.append(Task(**task_track_default.__dict__))
        else:
            TaskList.append(Task(**task_rescue_default.__dict__))
        
        TaskList[idx_task].task_id = idx_task
        TaskList[idx_task].x = random.uniform(WorldInfoInput.limit_x[0], WorldInfoInput.limit_x[1])
        TaskList[idx_task].y = random.uniform(WorldInfoInput.limit_y[0], WorldInfoInput.limit_y[1])
        TaskList[idx_task].z = random.uniform(WorldInfoInput.limit_z[0], WorldInfoInput.limit_z[1])
        TaskList[idx_task].start_time = random.uniform(0, 100)
        TaskList[idx_task].end_time = TaskList[idx_task].start_time + TaskList[idx_task].duration

    for n in range(0,num_tasks):
        print("Task " + str(n))
        print(str(TaskList[n].x)+", "+str(TaskList[n].y)+", "+str(TaskList[n].z))
        print(str(TaskList[n].start_time)+" - "+str(TaskList[n].end_time))
    for m in range(0,num_agents):
        print("Agent " + str(n))
        print(str(AgentList[m].x)+", "+str(AgentList[m].y)+", "+str(AgentList[m].z))



    # # here is different than original CBBA codes
    # # create random agents (quad only)
    # for idx_agent in range(0, num_agents):
    #     AgentList.append(agent_quad_default)
    #     AgentList[idx_agent].agent_id = idx_agent
    #     AgentList[idx_agent].x = random.randint(WorldInfoInput.limit_x[0], WorldInfoInput.limit_x[1])
    #     AgentList[idx_agent].y = random.randint(WorldInfoInput.limit_y[0], WorldInfoInput.limit_y[1])
    #     AgentList[idx_agent].z = 0

    # # create random tasks (track only)
    # for idx_task in range(0, num_tasks):
    #     TaskList.append(task_track_default)
    #     TaskList[idx_task].task_id = idx_task
    #     TaskList[idx_task].x = random.randint(WorldInfoInput.limit_x[0], WorldInfoInput.limit_x[1])
    #     TaskList[idx_task].y = random.randint(WorldInfoInput.limit_y[0], WorldInfoInput.limit_y[1])
    #     TaskList[idx_task].z = 0


    return AgentList, TaskList


def remove_from_list(self, list_input: list, index: int):
    """
    Remove item from list at location specified by index, then append -1 at the end

    Example:
        list_input = [0, 1, 2, 3, 4]
        index = 2
        list_output = remove_from_list(list_input, index)
        list_output = [0, 1, 3, 4, -1]
    """

    list_output = ((-1) * np.ones((1, len(list_input)))).flatten()
    list_output[0 : index] = np.array(list_input[0 : index])
    list_output[index : -1] = np.array(list_input[index+1:])

    return list_output.tolist()


def insert_in_list(self, list_input: list, value: float, index: int):
    """
    Insert value into list at location specified by index, and delete the last one of original list

    Example:
        list_input = [0, 1, 2, 3, 4]
        value = 100
        index = 2
        list_output = insert_in_list(list_input, value, index)
        list_output = [0, 1, 100, 2, 3]
    """

    list_output = ((-1) * np.ones((1, len(list_input)))).flatten()
    list_output[0 : index] = np.array(list_input[0 : index])
    list_output[index] = value
    list_output[index+1:] = np.array(list_input[index:-1])

    return list_output.tolist()