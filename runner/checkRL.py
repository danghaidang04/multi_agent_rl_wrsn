# For file manipulation
import sys
import os

from more_itertools.more import time_limited

# Include the parent directory in the path to import the modules in the parent directory or subdirectories
# sys.path: A list of strings that specifies the search path for modules. Initialized from the environment variable PYTHONPATH, plus an installation-dependent default.
# os.path.dirname: Return the directory name of pathname path.
# The line bellow is necessary because it prevents some bugs when running the code from the terminal, but it is not necessary when running from the IDE.
# If you run the code from the runner folder, it may occur some errors with no module found.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# For testing only
# sys.path.remove(os.path.dirname(os.path.dirname(__file__)))
# print(sys.path)

# For plotting
import seaborn as sns
import matplotlib.pyplot as plt
from utils import draw_heatmap_state

# For data manipulation
import pandas as pd
import numpy as np

# For environment and agent
from controller.random.RandomController import RandomController
from rl_env.WRSN import WRSN


def log(net, mcs):
    """
        Logs the current simulation time every 100 units of time.

        Parameters:
        - net: The network object that contains the simulation environment.
        - mcs: The agents of the network.

        How to use:
        - Pass the network object (`net`) and any additional required arguments (`mcs`).
        - Ensure that `net.env` has attributes `now` (current simulation time) and `timeout`.

        Note:
        Do not modify the core code; this function is designed to yield timeouts.
        """
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        if net.env.now % 100 == 0:
            print("Global time:",net.env.now,"units")
        # yield turns a function into a generator, which can be iterated over.
        yield net.env.timeout(1.0)

# Initialize the network and the controller
network = WRSN(scenario_path="../physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="../physical_env/mc/mc_types/default.yaml"
               ,num_agent=3, map_size=100,density_map=True)
controller = RandomController()

# Reset the network
request = network.reset()

# Print the active targets
for id, _ in enumerate(network.net.targets_active):
    if _ == 0:
        print(id)

# Start the logging process, this code is not necessary for the simulation
network.env.process(log(network.net, network.agents))

# Agent makes actions until the end of the episode
# Remove the detailed_rewards after the action
is_time_exceed = False
time_limit = 1000
while not request["terminal"] :
    if network.net.env.now >= time_limit:
        print("Time exceed at:", network.net.env.now, "units")
        is_time_exceed = True
        break
    # print the current time
    print("The agent takes action at", network.net.env.now, "units")
    print(request["agent_id"], request["action"], request["terminal"])
    action = controller.make_action(request["agent_id"], request["state"], request["info"], network)
    request = network.step(request["agent_id"], action)


# Print the final time
if not is_time_exceed:
    print("The simulation ends due to target disconnection at time step:", network.net.env.now)