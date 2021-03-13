# CBBA-Python
This is a Python implementation of CBBA (Consensus-Based Bundle Algorithm).

You can see more details about CBBA from these papers.

* [Choi, H.-L., Brunet, L., and How, J. P., “Consensus-Based Decentralized Auctions for Robust Task Allocation,” IEEE Transactions on Robotics, vol. 25, Aug. 2009, pp. 912–926.](https://ieeexplore.ieee.org/abstract/document/5072249?casa_token=zYvs9usD3FYAAAAA:jz0SmSso6T5l107pHGJgIQhVNP3S4NEnnIPi6sRC--8aealzVFinApRitUzhISlprmsPjcr3)

* [Brunet, L., Choi, H.-L., and How, J. P., “Consensus-Based Auction Approaches for Decentralized Task Assignment,” AIAA Guidance, Navigation, and Control Conference (GNC), Honolulu, HI: 2008.](https://arc.aiaa.org/doi/abs/10.2514/6.2008-6839)

Require:
Python >= 3.7

This repo has been tested with:
* Python 3.9.1, macOS 11.2.1, numpy 1.20.1, matplotlib 3.3.4
* python 3.8.5, Ubuntu 20.04.2 LTS, numpy 1.20.1, matplotlib 3.3.4


Dependencies
============
For Python:
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)

```
$ pip3 install numpy matplotlib
```


Usage
=====

The parameters for Tasks and Agents are written in a configuration json file.
* ["AGENT_TYPES"]: agent type.
* ["TASK_TYPES"]: task type. The i-th task type is associated with the i-th agent type.
* ["AGENT_DEFAULT"]["NOM_VELOCITY"]: the average speed of agent [m/s].
* ["AGENT_DEFAULT"]["FUEL"]: the traveling penalty/cost of agent.
* ["TASK_DEFAULT"]["TASK_VALUE"]: the value/reward of task. With larger value, the task is more important than others.
* ["TASK_DEFAULT"]["START_TIME"]: the starting timestamp of task [sec].
* ["TASK_DEFAULT"]["END_TIME"]: the enging timestamp of task [sec].
* ["TASK_DEFAULT"]["DURATION"]: the duration/time window of task [sec]. An agent needs to arrives at a task before the starting time and stays there until the ending time to be counted as complete a task.

An example `config_example_01.json`:
```json
{
    "AGENT_TYPES": ["quad", "car"],
    "TASK_TYPES": ["track", "rescue"],

    "QUAD_DEFAULT": {
        "NOM_VELOCITY": 2,
        "FUEL": 3
    },

    "CAR_DEFAULT": {
        "NOM_VELOCITY": 2,
        "FUEL": 3
    },
    
    "TRACK_DEFAULT": {
        "TASK_VALUE": 100,
        "START_TIME": 0,
        "END_TIME": 150,
        "DURATION": 15
    },

    "RESCUE_DEFAULT": {
        "TASK_VALUE": 100,
        "START_TIME": 0,
        "END_TIME": 150,
        "DURATION": 5
    }
}
```

The algorithm's main function is `CBBA.solve()`. An example is shown below.
```python
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from CBBA import CBBA
from Agent import Agent
from Task import Task
from WorldInfo import WorldInfo
import HelperLibrary as hp


if __name__ == "__main__":
    # a json configuration file
    config_file_name = "config_example_01.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # create a dataclass WorldInfo, each list is [min, max] coordinates for x,y,z axis
    WorldInfoInput = WorldInfo([-2.0,2.5], [-1.5,5.5], [0.0,0.0])

    num_agents = 5 # number of agents
    num_tasks = 10 # number of tasks
    max_depth = num_tasks # the maximum iteration number
    # randomly generate a list of dataclass Agent and Task
    # details in lib/HelperLibrary.py
    AgentList, TaskList = hp.create_agents_and_tasks(num_agents, num_tasks, WorldInfoInput, config_data)

    # create a CBBA solver with configuration data
    CBBA_solver = CBBA(config_data)

    # solve, time_window_flag has no effect yet, you can ignore it for now.

    # path_list is a 2D list, i-th sub-list is the task exploration order of Agent-i.
    # e.g. path_list = [[0,4,3,1], [2,5]] means Agent-0 visits Task 0,4,3,1, and Agent-1 visits Task 2,5

    # times_list is a 2D list, i-th sub-list is the task beginning time of Agent-i's tasks.
    path_list, times_list = CBBA_solver.solve(AgentList, TaskList, WorldInfoInput, max_depth, time_window_flag=True)

    # plot results
    CBBA_solver.plot_assignment()
    # you have to add this line in your script because plt.show() incide CBBA.plot_assignment() is in unblock mode
    plt.show()
```

Example
=======

A simple example with task time window is `test/test_cbba_example_01.py`.
```
$ cd <MAIN_DIRECTORY>
$ python3 test/test_cbba_example_01.py
```
The task assignment for each agent is stored as a 2D list `path_list` (the return variable of `CBBA.solve()`). The result visualization is shown below.
![A simple example with task time window a](/doc/1_a.png)
![A simple example with task time window b](/doc/1_b.png)


Another example with task time window (but the task duration is zero) is `test/test_cbba_example_02.py`.
```
$ cd <MAIN_DIRECTORY>
$ python3 test/test_cbba_example_02.py
```
The task assignment for each agent is stored as a 2D list `path_list` (the return variable of `CBBA.solve()`). The result visualization is shown below.
![A simple example with task time window 2](/doc/2.png)


<!-- An example without task time window is `test/test_cbba_example_03.py`.
```
$ cd <MAIN_DIRECTORY>
$ python3 test/test_cbba_example_03.py
```
The result visualization is shown below.
![A simple example without task time window](/doc/3.png) -->