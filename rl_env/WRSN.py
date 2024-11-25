import yaml
import copy

# Import the necessary modules from OpenAI Gym
import gym
from gym import spaces # spaces.Box is used to define the continuous observation and action spaces
# parameters of spaces.Box: low, high, shape, dtype

# For file manipulation
import sys
import os

# For number and visualization
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the necessary modules from the physical environment
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger

# Import scipy modules for the density map
from scipy.spatial.distance import euclidean
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize

def func(x, hX): 
    km = x ** 2 / (-2 * hX ** 2) 
    return np.exp(km)

# The class WRSN is inherited from the gym.Env class, which is the base class for all environments in OpenAI Gym.
class WRSN(gym.Env):
    def __init__(self, scenario_path, agent_type_path, num_agent, map_size=100, warm_up_time = 100, density_map=False):
        # Used to load the network configuration (nodes, targets, etc.) from a YAML file.
        self.scenario_io = NetworkIO(scenario_path)
        # Physical parameters of the mobile chargers (agents), loaded from another YAML file (agent_type_path).
        with open(agent_type_path, "r") as file:
            self.agent_phy_para = yaml.safe_load(file)
        # The number of mobile chargers in the environment.
        self.num_agent = num_agent
        # The size of the environment map (used for visualization and grid-based state representations).
        self.map_size = map_size
        # Boolean flag to enable/disable density map functionality.
        self.density_map = density_map
        # The initial simulation time for warming up the network.
        self.warm_up_time = warm_up_time
        # The epsilon value used for dividing by zero and handling numerical errors.
        self.epsilon = 1e-9
        # Defines the observation space for the environment in the form of a 4-channel grid.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4, self.map_size, self.map_size,), dtype=np.float64)
        # Defines the macro-action space for the environment as a continuous space with 3 dimensions, which is the location (x, y) and charging time.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64)
        # Create a list of size num_agent to store the latest action in the previous phase.
        self.agents_input_action = [None for _ in range(num_agent)]
        # Create a list of size num_agent to store the process of each agent, including moving, charging, ...
        self.agents_process = [None for _ in range(num_agent)]
        # Create a list of size num_agent to store the current action in this phase.
        self.agents_action = [None for _ in range(num_agent)]
        # Create a list of size num_agent to store the previous state of each agent.
        self.agents_prev_state = [None for _ in range(num_agent)]
        # Create a list of size num_agent to store the previous fitness of each agent.
        self.agents_prev_fitness = [None for _ in range(num_agent)]
        # Initializes a list to store the exclusive rewards for each agent.
        self.agents_exclusive_reward = [0 for _ in range(num_agent)]
        # Reset the environment, inherited from the gym.Env class. This is an abstract method that must be implemented.
        self.reset()

    # This function reinitialized the environment and the network, and returns the initial state of the environment.
    # It is called at the beginning of each episode, so we can intialize some dynamic attributes here
    # The static one can be used in the constructor.
    def reset(self):
        # Creates the network (self.net) and simulation environment (self.env) based on the scenario defined in the YAML file.
        # self.env: The simulation environment where processes (like charging, moving) are run.
        # self.net: The network structure.
        self.env, self.net = self.scenario_io.makeNetwork() # The makeNetwork method is defined in the NetworkIO class.

        # Initializes the network processes (like charging, moving) and the reward update process.
        self.net_process = self.env.process(self.net.operate()) & self.env.process(self.update_reward()) # Doing at the same time

        # Creates a list of agents (mobile chargers), initializing each one with:
        # self.net.baseStation.location: The starting location of the base station.
        # self.agent_phy_para: Physical parameters for the agents (e.g., speed, charging range).
        self.agents = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.agent_phy_para) for _ in range(self.num_agent)]
        for id, agent in enumerate(self.agents):
            agent.env = self.env
            agent.net = self.net
            agent.id = id
            agent.cur_phy_action = [self.net.baseStation.location[0], self.net.baseStation.location[1], 0]

        # Initializes the maximum moving time and charging time for the agents.
        self.moving_time_max = (euclidean(np.array([self.net.frame[0], self.net.frame[2]]), np.array([self.net.frame[1], self.net.frame[3]]))) / self.agent_phy_para["velocity"]
        self.charging_time_max = (self.scenario_io.node_phy_spe["capacity"] - self.scenario_io.node_phy_spe["threshold"]) / (self.agent_phy_para["alpha"] / (self.agent_phy_para["beta"] ** 2))

        # Initializes the average number of nodes in the charging range of the agents.
        self.avg_nodes_agent = (self.net.nodes_density * np.pi * (self.agent_phy_para["charging_range"] ** 2))

        # Warm-Up the Environment
        self.env.run(until=self.warm_up_time)

        # Check Network Status
        if self.net.alive == 1:
            tmp_terminal = False
        else:
            tmp_terminal = True

        # Computes the overall fitness of the network (e.g., based on node energy levels).
        tmp_fitness = self.get_network_fitness()

        # Initializes the previous state, action, and fitness for each agent.
        for id, agent in enumerate(self.agents):
            self.agents_prev_state[id] = self.get_state(agent.id)
            self.agents_action[id] = np.reshape(np.append(self.down_mapping(copy.deepcopy(self.net.baseStation.location)), 0), (3,))
            self.agents_process[id] = self.env.process(self.agents[id].operate_step(copy.deepcopy(agent.cur_phy_action)))
            self.agents_prev_fitness[id] = tmp_fitness   
            self.agents_exclusive_reward[id] = 0.0  

        # Check Agent Action Validity
        for id, agent in enumerate(self.agents):
            # If the agent is already at the target location and not charging, this is the one that is prefered to work first
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < self.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id":id, 
                        "prev_state": self.agents_prev_state[id],
                        "input_action": self.agents_input_action[id],
                        "action":self.agents_action[id], 
                        "reward": 0.0,
                        "state": self.agents_prev_state[id],
                        "terminal":tmp_terminal,
                        "info": [self.net, self.agents]}
        # If no agent is at the target location
        return {"agent_id":None, 
                "prev_state": None,
                "input_action": None,
                "action": None,
                "reward": None,
                "state": None,
                "terminal":tmp_terminal,
                "info": [self.net, self.agents]}
    
    # Converts a global location (real-world coordinates) into a normalized space (scaled to [0, 1]).
    def down_mapping(self, location):
        return np.array([(location[0] - self.net.frame[0]) / (self.net.frame[1] - self.net.frame[0])
                        ,(location[1] - self.net.frame[2]) / (self.net.frame[3] - self.net.frame[2])])

    # Converts a normalized location (scaled to [0, 1]) back into the real-world coordinate system.
    def up_mapping(self, down_map):
        return np.array([down_map[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0]
                         , down_map[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2]])

    # Transforms an agent's action from a normalized representation into real-world coordinates,
    # Including charging time scaling.
    def translate(self, agent_id, action):
        return np.array([action[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0],
                action[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2],
                self.charging_time_max * action[2]])

    # To calculate and distribute exclusive rewards to agents for charging nodes in the network.
    def update_reward(self):
        while True:
            # Compute the priority for each node based on its energy status:
            priority = np.array([(node.energyCS / (node.energy - node.threshold + self.epsilon)) if node.status != 0 else 0 for node in self.net.listNodes])

            # Standardize the priorities
            mean = np.mean(priority)
            std = np.std(priority)
            if std == 0:
                std = self.epsilon
            priority = (priority - mean) / std
            priority = np.exp(priority)

            # Normalize the priorities
            tmp_sum = np.sum(priority)
            if tmp_sum == 0:
                tmp_sum = self.epsilon
            priority = priority / tmp_sum

            # Loop Through Active Agents, calculate the exclusive reward for each agent (only the charging one)
            for agent in self.agents:
                if agent.status == 0:
                    continue
                if agent.cur_action_type == "charging":
                    incentive = 0
                    for node in agent.connected_nodes:
                        if node.status == 1:
                            chargingRate = (agent.alpha) / ((euclidean(node.location, agent.location) + agent.beta) ** 2)
                            energy_no_charge = min(node.energy - node.energyCS, node.threshold)
                            energy_with_charge = max(node.energy - node.energyCS + chargingRate, node.capacity)
                            incentive += priority[node.id] * (energy_with_charge - energy_no_charge) / ((agent.alpha) / (agent.beta ** 2))
                    self.agents_exclusive_reward[agent.id] += incentive
            yield self.env.timeout(1.0)
            
    # To calculate the state of the environment based on the current network status.
    def get_state(self, agent_id):
        agent = self.agents[agent_id]
        unit = 1.0 / self.map_size
        x = np.arange(unit / 2, 1.0,  unit)
        y = np.arange(unit / 2, 1.0,  unit)
        yy, xx = np.meshgrid(x, y)

        # Channel 1: Node Priorities
        map_1 = np.zeros_like(xx)
        for node in self.net.listNodes:
            if node.status == 0: 
                continue
            coor = self.down_mapping(node.location)
            xx_coor = xx - coor[0]
            yy_coor = yy - coor[1]
            hX = agent.chargingRange / (self.net.frame[1] - self.net.frame[0])
            hY = agent.chargingRange / (self.net.frame[3] - self.net.frame[2])
            pdf = ((node.energyCS / (agent.alpha / (agent.beta ** 2))) / ((node.energy - node.threshold) / (node.capacity - node.threshold))) * func(xx_coor, hX) * func(yy_coor, hY)
            map_1 += pdf
        # Channel 2: Agent Energy Distribution
        map_2 = np.zeros_like(xx)
        coor = self.down_mapping(agent.location)
        xx_coor = xx - coor[0]
        yy_coor = yy - coor[1]
        tmp = min((self.net.frame[3] - self.net.frame[2]), (self.net.frame[1] - self.net.frame[0]))
        hX =  0.5 * tmp / (self.net.frame[1] - self.net.frame[0])
        hY =  0.5 * tmp / (self.net.frame[3] - self.net.frame[2])
        pdf = (agent.energy / agent.capacity) * func(xx_coor, hX) * func(yy_coor, hY)
        map_2 += pdf

        # Channel 3: Charging Actions of Other Agents
        map_3 = np.zeros_like(xx)
        for another in self.agents:
            if another.id == agent.id:
                continue
            if another.cur_action_type == "moving":
                continue
            coor = self.down_mapping([another.cur_phy_action[0], another.cur_phy_action[1]])
            xx_coor = xx - coor[0]
            yy_coor = yy - coor[1]
            hX = another.chargingRange / (self.net.frame[1] - self.net.frame[0])
            hY = another.chargingRange / (self.net.frame[3] - self.net.frame[2])
            pdf = (another.cur_phy_action[2] / self.charging_time_max) * func(xx_coor, hX) * func(yy_coor, hY)
            map_3 += pdf

        # Channel 4: Movement Actions of Other Agents
        map_4 = np.zeros_like(xx)
        for another in self.agents:
            if another.id == agent.id:
                continue
            if another.cur_action_type == "charging":
                continue
            coor = self.down_mapping([another.cur_phy_action[0], another.cur_phy_action[1]])
            xx_coor = xx - coor[0]
            yy_coor = yy - coor[1]
            hX = another.chargingRange / (self.net.frame[1] - self.net.frame[0])
            hY = another.chargingRange / (self.net.frame[3] - self.net.frame[2])
            pdf = func(xx_coor, hX) * func(yy_coor, hY) * (euclidean(another.location, np.array([another.cur_phy_action[0], agent.cur_phy_action[1]])) / another.velocity) / self.moving_time_max
            map_4 += pdf
        return np.stack((map_1, map_2, map_3, map_4))

    # The function evaluates the fitness of the network
    def get_network_fitness(self):
        node_t = [-1 for node in self.net.listNodes]
        tmp1 = []
        tmp2 = []
        for node in self.net.baseStation.direct_nodes:
            if node.status == 1:
                tmp1.append(node)
                if node.energyCS == 0:
                    node_t[node.id] = float("inf")
                else:
                    node_t[node.id] = (node.energy - node.threshold) / (node.energyCS)
        while True:
            if len(tmp1) == 0:
                break
            for node in tmp1:
                for neighbor in node.neighbors:
                    if neighbor.status != 1:
                        continue
                    if neighbor.energyCS == 0:
                        neighborLT = float("inf")
                    else:
                        neighborLT = (neighbor.energy - neighbor.threshold) / (neighbor.energyCS)
                    if  node_t[neighbor.id] == -1 or (node_t[node.id] > node_t[neighbor.id] and neighborLT > node_t[neighbor.id]):
                        tmp2.append(neighbor)
                        node_t[neighbor.id] = min(neighborLT, node_t[node.id])

            tmp1 = tmp2[:]
            tmp2.clear()
        target_t = [0 for target in self.net.listTargets]
        for node in self.net.listNodes:
            for target in node.listTargets:
                target_t[target.id] = max(target_t[target.id], node_t[node.id])
        return np.array(target_t)

    # The overall (total) reward function for the environment
    def get_reward(self, agent_id):
        prev_fitness = self.agents_prev_fitness[agent_id]
        fitness = self.get_network_fitness()
        term_all = np.min(fitness) - np.min(prev_fitness)
        term_exclusive = self.agents_exclusive_reward[agent_id] / self.avg_nodes_agent
        return ((term_all * 0.8 + 0.2 * term_exclusive) / (self.charging_time_max + self.moving_time_max))

    # Map the action matrix to actual action
    def density_map_to_action(self, dmap, id):
        net = self.net
        agent = self.agents[id]
        unit = 1.0 / self.map_size
        
        max_index = np.unravel_index(np.argmax(dmap), dmap.shape)
        
        lower_bound = self.up_mapping([(max_index[0] + 0.5 ) * unit - agent.chargingRange / (net.frame[1] - net.frame[0]), (max_index[1] + 0.5) * unit - agent.chargingRange / (net.frame[3] - net.frame[2])])
        upper_bound = self.up_mapping([(max_index[0] + 0.5 ) * unit + agent.chargingRange / (net.frame[1] - net.frame[0]), (max_index[1] + 0.5) * unit + agent.chargingRange / (net.frame[3] - net.frame[2])])
        bounds = [(lower_bound[0], upper_bound[0]), (lower_bound[1], upper_bound[1])]
        def objective_function(loc):
            loc = np.array(loc)
            res = 0
            for node in net.listNodes:
                if node.status == 0:
                    continue
                res += int(euclidean(loc, node.location) <= agent.chargingRange) * (node.energyCS / (node.energy - node.threshold)) * agent.alpha / ((euclidean(loc, node.location) + agent.beta) ** 2)
            #print(loc, -res)
            return -res
        
        result = minimize(objective_function, [(lower_bound[0] + upper_bound[0]) / 2, (lower_bound[1] + upper_bound[1]) / 2], bounds=bounds, method='L-BFGS-B')
        #print(result)
        '''
        for node in net.listNodes:
            if euclidean(result.x, node.location) <= agent.chargingRange:
                print(node.id, euclidean(result.x, node.location), node.energyCS)
        node_x = [node.location[0] for node in net.listNodes]
        node_y = [node.location[1] for node in net.listNodes]
        target_x = [target.location[0] for target in net.listTargets]
        target_y = [target.location[1] for target in net.listTargets]
        plt.scatter(np.array(node_x), np.array(node_y))
        plt.scatter(np.array([net.baseStation.location[0]]), np.array([net.baseStation.location[1]]), c="red")
        plt.scatter(np.array(target_x), np.array(target_y), c="green")
        plt.scatter(np.array([result.x[0]]), np.array([result.x[1]]), c="purple")
        # Draw the rectangle boundaries
        lower_x = bounds[0][0]
        upper_x = bounds[0][1]
        lower_y = bounds[1][0]
        upper_y = bounds[1][1]
        # Draw the rectangle boundaries using plt.plot
        plt.plot([lower_x, upper_x], [lower_y, lower_y], color='red')  # Bottom side
        plt.plot([lower_x, upper_x], [upper_y, upper_y], color='red')  # Top side
        plt.plot([lower_x, lower_x], [lower_y, upper_y], color='red')  # Left side
        plt.plot([upper_x, upper_x], [lower_y, upper_y], color='red')  # Right side
        # Show the plot
        plt.show()
        '''
        prob = np.copy(dmap)
        flat_prob = prob.flatten()
        # Calculate the threshold value
        threshold = np.percentile(flat_prob, 99.9)
        # Set elements below the threshold to zero
        flat_prob[flat_prob < threshold] = 0
        # Reshape the flattened array back to the original shape
        prob = flat_prob.reshape(prob.shape)

        prob = prob / np.sum(prob)
        tmp_loc = self.down_mapping(np.array(result.x))
        return np.array([tmp_loc[0], tmp_loc[1], prob[max_index[0]][max_index[1]]])
    
    def step(self, agent_id, input_action):
        if agent_id is not None:
            # Process the Action
            action = np.array(input_action)
            self.agents_input_action[agent_id] = action.copy()

            # Normalize the Action (if Using a Density Map)
            if self.density_map:
                if not (np.all((action >= 0) & (action <= 1)) and np.isclose(np.sum(action), 1)):
                    action = np.exp(action)
                    action = action / (np.sum(action) + self.epsilon)
                action = self.density_map_to_action(action, agent_id)

            # Clip the Action to the Action Space
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # Update Agent’s Action and Start the Process
            self.agents_action[agent_id] = action
            self.agents_process[agent_id] = self.env.process(self.agents[agent_id].operate_step(self.translate(agent_id, action)))

            # Update Agent’s State and Rewards
            self.agents_prev_state[agent_id] = self.get_state(agent_id)
            self.agents_prev_fitness[agent_id] = self.get_network_fitness()
            self.agents_exclusive_reward[agent_id] = 0

        # Combine All Processes
        general_process = self.net_process
        for id, agent in enumerate(self.agents):
            if agent.status != 0:
                general_process = general_process | self.agents_process[id]

        # Run Simulation
        self.env.run(until=general_process)

        # Check if the Network is Alive
        if self.net.alive == 0:
            return {"agent_id":None, 
                    "prev_state": None,
                    "input_action": None,
                    "action":None, 
                    "reward": None,
                    "state": None,
                    "terminal":True,
                    "info": [self.net, self.agents]}

        # Find the Next Agent to Act
        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < self.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id": id, 
                        "prev_state": self.agents_prev_state[id],
                        "input_action":self.agents_input_action[id], 
                        "action":self.agents_action[id], 
                        "reward": self.get_reward(id),
                        "state": self.get_state(id), 
                        "terminal": False,
                        "info": [self.net, self.agents]}
