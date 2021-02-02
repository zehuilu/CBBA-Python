#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import numpy as np
from dataclasses import dataclass
from CBBA import CBBA
from Agent import Agent
from Task import Task


def func_test(a: dataclass):
    print(a.agent_id)


if __name__ == "__main__":
    CBBA_1 = CBBA(num_agents=3, num_tasks=5)
    print(CBBA_1.compatibility_mat)

    AAA = Agent()
    TTT = Task()

    print(AAA.agent_id)
    AAA.agent_id = 5
    print(AAA.agent_id)

    func_test(AAA)

    a=Agent()
    b=Agent()
    c=Agent()
    d=Agent()

    AgentList = [a,b,c,d]
    func_test(AgentList[3])
