# CBBA-Python
This is a Python implementation of CBBA.

Require:
Python >= 3.7


This repo has been tested with:
* Python 3.9.1, macOS 11.2.1


Dependencies
============
For Python:
* [numpy](https://numpy.org/).
* [matplotlib](https://matplotlib.org/).

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
![A simple example with task time window](/doc/with_time_window.png)



Another example without task time window is `test/test_cbba_example_02.py`.
```
$ cd <MAIN_DIRECTORY>
$ python3 test/test_cbba_example_02.py
```
The result visualization is shown below.
![A simple example with task time window](/doc/without_time_window.png)