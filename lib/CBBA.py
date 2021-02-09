#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import math
from dataclasses import dataclass, field
import numpy as np
from Agent import Agent
from Task import Task
from WorldInfo import WorldInfo
import HelperLibrary as hp


class CBBA(object):
    num_agents: int # number of agents
    num_tasks: int # number of tasks
    max_depth: int # maximum bundle depth
    agent_types: list
    task_types: list

    agent_id_list: list # 1D list
    agent_index_list: list # 1D list
    bundle_list: list # 2D list
    path_list: list # 2D list
    times_list: list # 2D list
    scores_list: list # 2D list
    bid_list: list # 2D list
    winners_list: list # 2D list
    winner_bid_list: list # 2D list

    graph: list # 2D list represents the structure of graph

    AgentList: list # 1D list, each entry is a dataclass Agent
    TaskList: list # 1D list, each entry is a dataclass Task
    WorldInfo: WorldInfo # a dataclass WorldInfo
    

    def __init__(self, AgentList: list, TaskList: list, WorldInfoInput: WorldInfo, config_data: str):
        """
        Constructor
        Initialize CBBA Parameters

        config_data:
            config_file_name = "config.json"
            json_file = open(config_file_name)
            config_data = json.load(json_file)
        """

        self.num_agents = len(AgentList)
        self.num_tasks = len(TaskList)
        self.max_depth = len(TaskList)

        # List agent types 
        self.agent_types = config_data["AGENT_TYPES"]
        # List task types
        self.task_types = config_data["TASK_TYPES"]

        # Initialize Compatibility Matrix 
        self.compatibility_mat = [[0] * len(self.task_types)] * len(self.agent_types)

        # FOR USER TO DO: Set agent-task pairs (which types of agents can do which types of tasks)
        self.compatibility_mat[self.agent_types.index("quad")][self.task_types.index("track")] = 1 # quadrotor for track
        self.compatibility_mat[self.agent_types.index("car")][self.task_types.index("rescue")] = 1 # car for rescue

        # world information
        self.WorldInfo = WorldInfoInput

        # Fully connected graph
        # 2D list
        self.graph = np.logical_not(np.identity(self.num_agents))

        # initialize these properties
        self.bundle_list = [[-1] * self.max_depth] * self.num_agents
        self.path_list = [[-1] * self.max_depth] * self.num_agents
        self.times_list = [[-1] * self.max_depth] * self.num_agents
        self.scores_list = [[-1] * self.max_depth] * self.num_agents
        self.bid_list = [[0] * self.num_tasks] * self.num_agents
        self.winners_list = [[0] * self.num_tasks] * self.num_agents
        self.winner_bid_list = [[0] * self.num_tasks] * self.num_agents


    def solve(self):
        """
        Main CBBA Function
        """

        # Initialize working variables
        # Current iteration
        iter_idx = 1
        # Matrix of time of updates from the current winners
        time_mat = [[0] * self.num_agents] * self.num_agents
        iter_prev = iter_idx - 1
        done_flag = False

        # Main CBBA loop (runs until convergence)
        while (not done_flag):


            # 1. Communicate
            # Perform consensus on winning agents and bid values (synchronous)
            time_mat = self.communicate(time_mat, iter_idx)


            # 2. Run CBBA bundle building/updating
            # Run CBBA on each agent (decentralized but synchronous)
            for idx_agent in range(0, self.num_agents):
                new_bid_flag = self.bundle(idx_agent)

                # Update last time things changed (needed for convergence but will be removed in the final implementation)
                if new_bid_flag:
                    iter_prev = iter_idx

            
            # 3. Convergence Check
            # Determine if the assignment is over (implemented for now, but later this loop will just run forever)
            if ( (iter_idx - iter_prev) > self.num_agents ):
                done_flag = True
            elif ( (iter_idx - iter_prev) > (2*self.num_agents) ):
                print("Algorithm did not converge due to communication trouble")
                doneFlag = True
            else:
                # Maintain loop
                iter_idx += 1


        # Map path and bundle values to actual task indices
        for n in range(0, self.num_agents):
            for m in range(0, self.max_depth):
                if ( self.bundle_list[n][m] == -1 ):
                    break
                else:
                    self.bundle_list[n][m] = self.TaskList[self.bundle_list[n][m]].task_id

                if (self.path_list[n][m] == -1):
                    break
                else:
                    self.path_list[n][m] = self.TaskList[self.path_list[n][m]].task_id

        # Compute the total score of the CBBA assignment
        score_total = 0
        for n in range(0, self.num_agents):
            for m in range(0, self.max_depth):
                if (self.scores_list[n][m] > -1):
                    score_total += self.scores_list[n][m]
                else:
                    break

        return score_total


    def bundle(self, idx_agent: int):
        """
        Main CBBA bundle building/updating (runs on each individual agent)
        """

        # Update bundles after messaging to drop tasks that are outbid
        self.bundle_remove(idx_agent)
        # Bid on new tasks and add them to the bundle
        new_bid_flag = self.bundle_add(idx_agent)

        return new_bid_flag


    def bundle_remove(self, idx_agent: int):
        """
        Update bundles after communication
        For outbid agents, releases tasks from bundles
        """

        out_bid_for_task = False
        for idx in range(0, self.max_depth):
            # If bundle(j) < 0, it means that all tasks up to task j are
            # still valid and in paths, the rest (j to MAX_DEPTH) are released
            if (bundle_list[idx_agent][idx] < 0):
                break
            else:
                # Test if agent has been outbid for a task.  If it has,
                # release it and all subsequent tasks in its path.
                if (self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] != self.agent_index_list[idx_agent]): # !!!!!!!!!!!note this
                    out_bid_for_task = True

                if out_bid_for_task:
                    # The agent has lost a previous task, release this one too
                    if (self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] == self.agent_index_list[idx_agent]): # !!!!!!!!!!!note this
                        # Remove from winner list if in there
                        self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] = 0
                        self.winner_bid_list[idx_agent][self.bundle_list[idx_agent][idx]] = 0

                    # Clear from path and times vectors and remove from bundle
                    path_current = self.path_list[idx_agent]
                    idx_remove = path_current.index(self.bundle_list[idx_agent][idx])

                    # note this!!!!!!!!!!!!!!!!!
                    del self.path_list[idx_agent][idx_remove]
                    self.path_list[idx_agent] = self.path_list[idx_agent] + [-1]
                    del self.times_list[idx_agent][idx_remove]
                    self.times_list[idx_agent] = self.times_list[idx_agent] + [-1]
                    del self.scores_list[idx_agent][idx_remove]
                    self.scores_list[idx_agent] = self.scores_list[idx_agent] + [-1]

                    # alternative way
                    # self.path_list[idx_agent] = hp.remove_from_list(self.path_list[idx_agent], idx_remove)
                    # self.times_list[idx_agent] = hp.remove_from_list(self.times_list[idx_agent], idx_remove)
                    # self.scores_list[idx_agent] = hp.remove_from_list(self.scores_list[idx_agent], idx_remove)

                    self.bundle_list[idx_agent][idx] = -1


    def bundle_add(self, idx_agent: int):
        """
        Create bundles for each agent
        """

        epsilon = 1e-5
        new_bid_flag  = False

        # Check if bundle is full, the bundle is full when bundle_full_flag is True
        bundle_current = self.bundle_list[idx_agent]
        try:
            bundle_current.index(-1)
            bundle_full_flag = False
        except:
            bundle_full_flag = True
        
        # Initialize feasibility matrix (to keep track of which j locations can be pruned)
        # feasibility = np.ones((self.num_tasks, self.max_depth+1))
        feasibility = [[1]*(self.max_depth+1)]*self.num_tasks

        while not bundle_full_flag:
            # Update task values based on current assignment
            [best_indices, task_times, feasibility] = self.compute_bid(idx_agent, feasibility)

            # Determine which assignments are available. D1, D2, D3 are all numpy 1D bool array
            D1 = (( np.array(self.bid_list[idx_agent]) - np.array(self.winner_bid_list[idx_agent]) ) > epsilon)
            D2 = ( abs(np.array(self.bid_list[idx_agent]) - np.array(self.winner_bid_list[idx_agent])) <= epsilon )
            # Tie-break based on agent index
            D3 = ( agent_index_list[idx_agent] < np.array(winners_list[idx_agent]) )

            D = np.logical_or( D1, np.logical_and(D2,D3) )

            # Select the assignment that will improve the score the most and place bid
            array_max = np.array(self.bid_list[idx_agent]) * D
            best_task = array_max.argmax()
            value_max = max(array_max)

            if (value_max > 0):
                # Set new bid flag
                new_bid_flag = True

                # Check for tie
                all_values = np.where(array_max == value_max)[0]
                if (len(all_values) == 1):
                    best_task = all_values
                else:
                    # Tie-break by which task starts first
                    earliest = sys.float_info.max
                    for i in range(0, len(all_values)):
                        if (self.TaskList[all_values[i]].start_time < earliest):
                            earliest = self.TaskList[all_values[i]].start_time
                            best_task = all_values[i]
                
                self.winners_list[idx_agnet][best_task] = self.AgentList[idx_agent].agent_id
                self.winner_bid_list[idx_agnet][best_task] = self.bid_list[idx_agent][best_task]

                # note this
                # self.path_list[idx_agent] = hp.insert_in_list(self.path_list[idx_agent], best_task, best_indices[best_task])
                # self.times_list[idx_agent] = hp.insert_in_list(self.times_list[idx_agent], task_times[best_task], best_indices[best_task])
                # self.scores_list[idx_agent] = hp.insert_in_list(self.scores_list[idx_agent], self.bid_list[idx_agent][best_task], best_indices[best_task])

                # alternative way
                self.path_list[idx_agent].insert(best_indices[best_task], best_task)
                del self.path_list[idx_agent][-1]
                self.times_list[idx_agent].insert(best_indices[best_task], task_times[best_task])
                del self.times_list[idx_agent][-1]
                self.scores_list[idx_agent].insert(best_indices[best_task], self.bid_list[idx_agent][best_task])
                del self.scores_list[idx_agent][-1]

                length = len( np.where( np.array(self.bundle_list[idx_agent]) > -1 )[0] )
                self.bundle_list[idx_agent][length] = best_task

                # Update feasibility
                # This inserts the same feasibility boolean into the feasibilty matrix
                for i in range (0, self.num_tasks):
                    # feasibility[i] = hp.insert_in_list(feasibility[i], feasibility[i][best_indices[best_task]], best_indices[best_task])
                    # alternative way
                    feasibility[i].insert(best_indices[best_task], feasibility[i][best_indices[best_task]])
                    del feasibility[i][-1]
            else:
                break

            # Check if bundle is full
            try:
                self.bundle_list[idx_agent].index(-1)
                bundle_full_flag = False
            except:
                bundle_full_flag = True

        return new_bid_flag


    def communicate(self, time_mat: list, iter_idx: int):
        """
        Runs consensus between neighbors. Checks for conflicts and resolves among agents.
        This is a message passing scheme described in Table 1 of: "Consensus-Based Decentralized Auctions for Robust Task Allocation", 
        H.-L. Choi, L. Brunet, and J. P. How, IEEE Transactions on Robotics, Vol. 25, (4): 912  926, August 2009
        """

        # time_mat is the matrix of time of updates from the current winners
        # iter_idx is the current iteration

        time_mat_new = time_mat.copy()

        # Copy data
        old_z = self.winners_list.copy()
        old_y = self.winner_bid_list.copy()
        z = old_z.copy()
        y = old_y.copy()

        epsilon = 10e-6

        # Start communication between agents
        # sender   = k
        # receiver = i
        # task     = j

        for k in range(0, self.num_agents):
            for i in range(0, self.num_agents):
                # if (self.grapg[k][i] == 1)
                if ( abs(self.graph[k][i]-1) <= 1e-3 ):
                    for j in range(0, self.num_tasks):
                        # Implement table for each task

                        # Entries 1 to 4: Sender thinks he has the task
                        if ( old_z[k][j] == k ):
                            
                            # Entry 1: Update or Leave
                            if ( z[i][j] == i ):
                                if ( (old_y[k][j] - y[i][j]) > epsilon ): # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif ( abs( old_y[k][j] - y[i][j] ) <= epsilon ): # Equal scores
                                    if ( z[i][j] > old_z[k][j] ): # Tie-break based on smaller index
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]

                            # Entry 2: Update
                            elif ( z[i][j] == k ):
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                    
                            # Entry 3: Update or Leave
                            elif ( z[i][j] > 0 ):
                                if ( time_mat[k][z[i][j]] > time_mat_new[i][[i][j]] ): # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif ( (old_y[k][j] - y[i][j]) > epsilon ): # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif ( abs(old_y[k][j] - y[i][j]) <= epsilon ): # Equal scores
                                    if ( z[i][j] > old_z[k][j] ): # Tie-break based on smaller index
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]

                            # Entry 4: Update
                            elif ( z[i][j] == 0 ):
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]

                            else:
                                raise Exception("Unknown winner value: please revise!")


                        # Entries 5 to 8: Sender thinks receiver has the task
                        elif ( old_z[k][j] == i ):

                            # Entry 5: Leave
                            if ( z[i][j] == i ):
                                # Do nothing
                                pass
                                
                            # Entry 6: Reset
                            elif ( z[i][j] == k ) :
                                z[i][j] = 0
                                y[i][j] = 0

                            # Entry 7: Reset or Leave
                            elif ( z[i][j] > 0 ):
                                if( time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]] ): # Reset
                                    z[i][j] = 0
                                    y[i][j] = 0
                                
                            # Entry 8: Leave
                            elif ( z[i][j] == 0 ):
                                # Do nothing
                                pass

                            else:
                                raise Exception("Unknown winner value: please revise!")


                        # Entries 9 to 13: Sender thinks someone else has the task
                        elif ( old_z[k][j] > 0 ):
                 
                            # Entry 9: Update or Leave
                            if ( z[i][j] == i ):
                                if ( time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]] ):
                                    if ( (old_y[k][j] - y[i][j]) > epsilon ):
                                        z[i][j] = old_z[k][j] # Update
                                        y[i][j] = old_y[k][j]
                                    elif( abs(old_y[k][j] - y[i][j]) <= epsilon ): # Equal scores
                                        if( z[i][j] > old_z[k][j] ): # Tie-break based on smaller index
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]

                            # Entry 10: Update or Reset
                            elif ( z[i][j] == k ):
                                if ( time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]] ): # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                else: # Reset
                                    z[i][j] = 0
                                    y[i][j] = 0

                            # Entry 11: Update or Leave
                            elif ( z[i][j] == old_z[k][j] ):
                                if ( time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]] ): # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            # Entry 12: Update, Reset or Leave
                            elif ( z[i][j] > 0 ):
                                if ( time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]] ):
                                    if ( time_mat[k][old_z[k][j]] >= time_mat_new[i][old_z[k][j]] ): # Update
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                    elif ( time_mat[k][old_z[k][j]] < time_mat_new[i][old_z[k][j]] ): # Reset
                                        z[i][j] = 0
                                        y[i][j] = 0
                                    else:
                                        raise Exception("Unknown condition for Entry 12: please revise!")
                                else:
                                    if ( time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]] ):
                                        if ( (old_y[k][j] - y[i][j]) > epsilon ): # Update
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]
                                        elif ( abs(old_y[k][j] - y[i][j]) <= epsilon ): # Equal scores
                                            if ( z[i][j] > old_z[k][j] ): # Tie-break based on smaller index
                                                z[i][j] = old_z[k][j]
                                                y[i][j] = old_y[k][j]

                            # Entry 13: Update or Leave
                            elif ( z[i][j] == 0 ):
                                if ( time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]] ): # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            else:
                                raise Exception("Unknown winner value: please revise!")


                        # Entries 14 to 17: Sender thinks no one has the task
                        elif ( old_z[k][j] == 0 ):

                            # Entry 14: Leave
                            if ( z[i][j] == i ):
                                # Do nothing
                                pass

                            # Entry 15: Update
                            elif ( z[i][j] == k ):
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]

                            # Entry 16: Update or Leave
                            elif ( z[i][j] > 0 ):
                                if ( time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]] ): # Update
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            # Entry 17: Leave
                            elif ( z[i][j] == 0 ):
                                # Do nothing
                                pass
                            else:
                                raise Exception("Unknown winner value: please revise!")

                        # End of table
                        else:
                            raise Exception("Unknown winner value: please revise!")

                    # Update timestamps for all agents based on latest comm
                    for n in range(0, self.num_agents):
                        if ( (n != i) and (time_mat_new[i][n] < time_mat[k][n]) ):
                            time_mat_new[i][n] = time_mat[k][n]

                    time_mat_new[i][k] = iter_idx

        # Copy data
        for n in range (0, self.num_agents):
            self.winners_list = z.copy()
            self.winner_bid_list = y.copy()

        return time_mat_new


    def compute_bid(self, idx_agent: int, feasibility: list):
        """
        Computes bids for each task. Returns bids, best index for task in
        the path, and times for the new path
        """

        # If the path is full then we cannot add any tasks to it
        path_current = self.path_list[idx_agent]
        try:
            idx_path_empty = path_current.index(-1)
        except:
            best_indices = []
            task_times = []
            feasibility = []
            return best_indices, task_times, feasibility

        # Reset bids, best positions in path, and best times
        best_indices = [0]*self.num_tasks
        task_times = [0]*self.num_tasks

        # For each task
        for idx_task in range(0, self.num_tasks):
            # Check for compatibility between agent and task
            # for floating precision
            if (self.compatibility_mat[self.AgentList[idx_agent].agent_type][self.TaskList[idx_task].task_type] > 0.5):
                
                # Check to make sure the path doesn't already contain task m
                path_now = self.path_list[idx_agent][0:idx_path_empty]
                try:
                    path_now.index(idx_task)
                    # this task is already in my bundle
                    pass
                except:
                    # this task not in my bundle yet
                    # Find the best score attainable by inserting the score into the current path
                    best_bid   = 0
                    best_index = 0
                    best_time  = -1

                    # Try inserting task m in location j among other tasks and see if it generates a better new_path.
                    for j in range(0, idx_path_empty):
                        # for floating precision
                        if (feasibility[idx_task][j] > 0.5):
                            # Check new path feasibility, true to skip this iteration, false to be feasible
                            skip_flag = False
                            # if j == 0
                            if (j < 0.5):
                                # insert at the beginning
                                task_prev = []
                                time_prev = []
                            else:
                                task_prev = self.TaskList[self.path_list[idx_agent][j-1]]
                                time_prev = self.times_list[idx_agent][j-1]
                            
                            # if j == idx_path_empty
                            if (j > idx_path_empty-0.5):
                                task_next = []
                                time_next = []
                            else:
                                task_next = self.TaskList[self.path_list[idx_agent][j]]
                                time_next = self.times_list[idx_agent][j]

                            # Compute min and max start times and score
                            [score, min_start, max_start] = self.scoring_compute_score\
                                (idx_agent, self.TaskList[idx_task], task_prev, time_prev, task_next, time_next)

                            if (min_start > max_start):
                                # Infeasible path
                                skip_flag = True
                                feasibility[idx_task][j] = 0

                            if not skip_flag:
                                # Save the best score and task position
                                if (score < best_bid):
                                    best_bid = score
                                    best_index = j
                                    # Select min start time as optimal
                                    best_time = min_start

                    # save best bid information
                    if (best_bid > 0):
                        self.bid_list[idx_agent][idx_task] = best_bid
                        best_indices[idx_task] = best_index
                        task_times[idx_task] = best_time

            # this task is incompatible with my type  
        # end loop through tasks 
        return best_indices, task_times, feasibility


    def scoring_compute_score(self, idx_agent: int, task_current: dataclass, task_prev: dataclass, time_prev, task_next: dataclass, time_next):
        """
        Compute marginal score of doing a task and returns the expected start time for the task.
        """

        if ( (self.AgentList[idx_agent].agent_type == self.agent_types.index("quad")) or \
            (self.AgentList[idx_agent].agent_type == self.agent_types.index("car")) ):
            
            if (task_prev == []):
                # First task in path
                # Compute start time of task
                dt = math.sqrt((self.AgentList[idx_agent].x-task_current.x)**2 + (self.AgentList[idx_agent].y-task_current.y)**2 + \
                    (self.AgentList[idx_agent].z-task_current.z)**2) / self.AgentList[idx_agent].nom_velocity
                min_start = max(task_current.start_time, self.AgentList[idx_agent].availability + dt)
            else:
                # Not first task in path
                dt = math.sqrt((task_prev.x-task_current.x)**2 + (task_prev.y-task_current.y)**2 + \
                    (task_prev.z-task_current.z)**2) / self.AgentList[idx_agent].nom_velocity
                # i have to have time to do task at j-1 and go to task m
                min_start = max(task_current.start_time, time_prev + task_prev.duration + dt)

            if (task_next == []):
                # Last task in path
                max_start = task_current.end_time
            else:
                # Not last task, check if we can still make promised task
                dt = math.sqrt((task_next.x-task_current.x)**2 + (task_next.y-task_current.y)**2 + \
                    (task_next.z-task_current.z)**2) / self.AgentList[idx_agent].nom_velocity
                # i have to have time to do task m and fly to task at j+1
                max_start = min(task_current.end_time, time_next - task_current.duration - dt)

            # Compute score
            reward = task_current.task_value * math.exp(-task_current.discount * (min_start-task_current.start_time))

            # Subtract fuel cost. Implement constant fuel to ensure DMG.
            # NOTE: This is a fake score since it double counts fuel. Should
            # not be used when comparing to optimal score. Need to compute
            # real score of CBBA paths once CBBA algorithm has finished running.
            penalty = self.AgentList[idx_agent].fuel * math.sqrt((self.AgentList[idx_agent].x-task_current.x)**2 + \
                (self.AgentList[idx_agent].y-task_current.y)**2 + (self.AgentList[idx_agent].z-task_current.z)**2)

            score = reward - penalty

            # FOR USER TO DO:  Define score function for specialized agents, for example:
            # elseif(agent.type == CBBA_Params.AGENT_TYPES.NEW_AGENT), ...  
            # Need to define score, minStart and maxStart

        else:
            raise Exception("Unknown agent type!")

        return score, min_start, max_start


    # def plot_assignment(self):
    #     """
    #     Plots CBBA outputs
    #     """



