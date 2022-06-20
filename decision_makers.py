
import numpy as np
import torch
import numpy as np
import os
#from network import policy_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class decision_maker:
    def __init__(self, int_Ex, include_budget, shutdown_cond, discrete_actions):
        self.int_Ex = int_Ex
        self.include_budget = include_budget
        self.exp_interval = np.random.randint(2,10,1)
        self.shutdown_cond = shutdown_cond
        self.discrete_actions = discrete_actions
        self.file_name =''#'params-hrl-infra_env-it(100000)-[19, 0].tar'#'params-hrl-infra_env-it(100000)-[16, 3].tar'
        if self.file_name !='':
            self.policy_nn = policy_net(action_size=2).to(device)
            self.policy_net = self.load_agent(self.file_name)

    # element level decision makers
    def agent_element_lvl_discrete(self, state):
        goal = 1 if (state[2] % self.exp_interval == 0 and state[2] != 0) else 0
        if state[0] > 70 and state[1] > -1.5 :
            goal = 0
        if self.include_budget: 
            if state[-1] < 0: 
                goal = 0
        return goal

    def agent_one_element_lvl(self, state):
        if self.discrete_actions:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif state[0] < 55 :
                goal = self.int_Ex[0][2,0]/75
            elif state[0] < 70 :
                goal = self.int_Ex[0][1,0]/75
            else:
                goal = self.int_Ex[0][0,0]/75
        else:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif state[0] < 55 and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[0][2,0]/75
            elif state[0] < 70 and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[0][1,0]/75
            elif state[1] < -1.5 and state[2] % self.exp_interval == 0 and state[2] != 0:
                goal = self.int_Ex[0][0,0]/75
            else :
                goal = 0
            # budget check
            if self.include_budget:
                if state[-1] < 0 :
                    goal = 0
        return goal

    # high level decision makers
    def agent_multi_element_lvl(self, state_element):
        goal_dim = 0
        local_change = np.max([state_element[4] - state_element[goal_dim], 0])
        total_change = state_element[5] * 75 * state_element[3]
        if state_element[5] > 0 :
            ratio = np.min([local_change / total_change, 1])
            goal = ratio * state_element[5] * state_element[3]
        else:
            goal = 0
        return goal

    def agent_high_lvl_discrete(self, state):
        if self.file_name != '':
            goal = self.evaluate_agent(state)
        else:
            goal = 1 if (state[2] % 1 == 0 and state[2] != 0) else 0 #self.exp_interval
            if state[0] > 60 and state[1] > -1.5 and state[3] < 10 :
                goal = 0
            if self.include_budget: 
                if state[-1] < 0: 
                    goal = 0
        return goal

    def agent_high_lvl(self, state):
        if self.discrete_actions:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif state[0] < 55 or state[3] > 30:
                goal = self.int_Ex[0][2,0]/75
            elif state[0] < 70 or state[3] > 15:
                goal = self.int_Ex[0][1,0]/75
            else:
                goal = self.int_Ex[0][0,0]/75
        else:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif (state[0] < 55 or state[3] > 30) and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[0][2,0]/75
            elif (state[0] < 70 or state[3] > 15) and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[0][1,0]/75
            elif state[1] < -1.5 and state[2] % self.exp_interval == 0 and state[2] != 0:
                goal = self.int_Ex[0][0,0]/75
            else :
                goal = 0
            # budget check
            if self.include_budget:
                if state[-1] < 0 :
                    goal = 0
        return goal
    
    def load_agent(self, filename):
        # load params file
        file_path = os.path.join(".", "model", filename)
        params_file = torch.load(file_path)
        # initialize network weights
        self.policy_nn.load_state_dict(params_file['policy_net'])
        return policy_net

    def evaluate_agent(self, state):
        action = torch.argmax(self.policy_nn(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        return action[0]
