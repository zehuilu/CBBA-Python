#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import math
import numpy as np


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