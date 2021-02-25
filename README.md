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

The parameters for Tasks and Agents are written in a configuration json file. An example is `config_example_01.json`.

The algorithm's main function is `CBBA.solve()`. An example is shown below.


Example
=======

A simple example with task time window is `test/test_cbba_example_01.py`.
```
$ cd <MAIN_DIRECTORY>
$ python3 test/test_cbba_example_01.py
```
The task assignment for each agent is stored as a 2D list `path_list` (the return variable of `CBBA.solve()`). The result visualization is shown below.
![A simple example with task time window](/doc/1.png)


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