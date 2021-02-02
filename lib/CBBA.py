#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
from dataclasses import dataclass
import numpy as np
from Agent import Agent
from Task import Task


class CBBA(object):
    num_agents: int # number of agents
    num_tasks: int # number of tasks
    max_depth: int # maximum bundle depth
    agent_types: list
    task_types: list

    agent_id_list: list
    agent_index_list: list
    bundle_list: list
    path_list: list
    times_list: list
    scores_list: list
    bid_list: list
    winners_list: list
    winner_bid_list: list
    


    def __init__(self, num_agents: int, num_tasks: int):
        """
        Constructor
        Initialize CBBA Parameters
        """

        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.max_depth = num_tasks

        # List agent types 
        self.agent_types = ["quad", "car"]
        # List task types
        self.task_types = ["track", "rescue"]

        # Initialize Compatibility Matrix 
        # self.compatibility_mat = [0*len(self.task_types)]*len(self.agent_types)
        self.compatibility_mat = np.zeros((len(self.agent_types), len(self.task_types)))

        # FOR USER TO DO: Set agent-task pairs (which types of agents can do which types of tasks)
        self.compatibility_mat[0, 0] = 1 # quadrotor for track
        self.compatibility_mat[1, 1] = 1 # car for rescue


    def solve(self, AgentList: list, TaskList: list):
        self.num_agents = len(AgentList)
        self.num_tasks = len(TaskList)
        self.max_depth = len(TaskList)


    def bundle_remove(self):
        """
        Update bundles after communication
        For outbid agents, releases tasks from bundles
        """

        out_bid_for_task = 0

        for idx in range(0, self.max_depth):
            # If bundle(j) < 0, it means that all tasks up to task j are
            # still valid and in paths, the rest (j to MAX_DEPTH) are released
            if (bundle_list[idx] < 0):
                break
            else:
                # Test if agent has been outbid for a task.  If it has,
                # release it and all subsequent tasks in its path.
                if (self.winners_list[self.bundle_list[idx]] != self.agent_index_list): # !!!!!!!!!!!note this
                    out_bid_for_task = 1

                if (out_bid_for_task):
                    # The agent has lost a previous task, release this one too
                    if (self.winners_list[self.bundle_list[idx]] == self.agent_index_list): # !!!!!!!!!!!note this
                        # Remove from winner list if in there
                        self.winners_list[self.bundle_list[idx]] = 0
                        self.winner_bid_list[self.bundle_list[idx]] = 0

                    # Clear from path and times vectors and remove from bundle
                    idx_remove = self.path_list.index(self.bundle_list[idx])

                    self.path_list = self.remove_from_list(self.path_list, idx_remove)
                    self.times_list = self.remove_from_list(self.times_list, idx_remove)
                    self.scores_list = self.remove_from_list(self.scores_list, idx_remove)

                    self.bundle_list[idx] = -1



    def remove_from_list(self, list_input: list, index: int):
        """
        Remove item from list at location specified by index
        """

        list_output = ((-1) * np.ones((1, len(list_input)))).flatten()

        list_output[0 : index] = np.array(list_input[0 : index])

        list_output[index : -1] = np.array(list_input[index+1:])

        return list_output