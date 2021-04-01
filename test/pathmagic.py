#!/usr/bin/env python3
import os
import sys


class context:
    def __enter__(self):
        # append the main directory of CBBA-Python to sys.path
        # assume the working directory is always the main directory of CBBA-Python
        sys.path.append(os.getcwd())
        print(os.getcwd())

    def __exit__(self, *args):
        pass
