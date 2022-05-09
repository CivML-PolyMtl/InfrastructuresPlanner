""" Infrastructures Planner ########################################################################################################

    Infrastructures Planner is a custom gym environment to plan interventions on infrastrucutres.
    Developed by: Zachary Hamida
    Email: zac.hamida@gmail.com
    Webpage: https://zachamida.github.io

"""

from gym import spaces
import numpy as np
import math as mt
import copy
import scipy.linalg as sp
import scipy.io as sio
import scipy.stats as stats
import scipy.special as sc
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from os.path import join as pjoin


class InfrastructuresPlanner:
    def __init__(self):
        self.reset()

    """ Initial Environment Run ########################################################################################################

        reset : reset the env

    """
    def reset(self):
        # Analyses level
        self.element_lvl = 1
        self.category_lvl = 0
        self.bridge_lvl = 0
        self.network_lvl = 0

        # State type
        self.deterministic_model = 0

        # action type
        self.discrete_actions = 0

        # budget
        self.include_budget = 0
        self.max_budget = 1e4
        self.min_budget = 1e3
        self.budget = np.random.uniform(self.min_budget, self.max_budget)

        # expert 
        self.expert_model = 0
        self.exp_interval = np.random.randint(2,10,1)

        # plotting    
        self.plotting = 0

        # actions and observations
        self.action_space = spaces.Discrete(2)
        high = np.array(
            [
                100,
                0,
                100,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                0,
                -20,
                0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.metadata = 'Infrastructures deterioration environment'
        # functions
        self.inv = np.linalg.pinv
        self.svd = np.linalg.svd
        # time props
        self.dt = 1        # time step size
        self.total_years = 100           # total time steps
        self.seed = np.random.seed
        # network props
        # Network[birdge #1(Category #1(#Elements), Category #2(#Elements)), 
        #         bridge #2(Category #1(#Elements), Category #2(#Elements))]
        self.net_data = np.array([
                                [[2],[3]],
                                [[2],[2]]
                                ])
        self.num_c = [ len(listElem) for listElem in self.net_data]
        self.num_b = self.net_data.shape[0]
        self.num_e = self.net_data
        self.initial_state = np.zeros([self.num_b, np.max(self.num_c), self.num_e.max(),3])

        # indicators
        self.cb = np.array(0)
        self.cc = np.array(0)
        self.ci = np.array(0)
        self.ec = np.array(0) # element index tracker can be in the state vector, also faciltates generateing a new inspector

        # inspection data
        self.max_cond = np.array(100)
        self.min_cond = np.array(25)
        self.y = np.nan * np.zeros([self.num_c[self.cb], np.max(self.num_e[self.cb,:, 0])])
        self.inspection_frq = np.random.randint(3,4,self.num_b)
        self.inspector_std = np.array(range(0,223))
        self.inspector_std = np.c_[self.inspector_std, np.random.uniform(1,6,223)]

        # kenimatic model
        self.F = np.array([1, 0, 0])
        self.A = np.array([[1, self.dt, (self.dt ** 2) / 2], [0, 1, self.dt], [0, 0, 1]])
        sigma_w = 0.005
        self.Q = sigma_w** 2 * np.array([[(self.dt ** 5) / 20, (self.dt ** 4) / 8, (self.dt ** 3) / 6],
                                      [(self.dt ** 4) / 8, (self.dt ** 3) / 3, (self.dt ** 2) / 2],
                                      [(self.dt ** 3) / 6, (self.dt ** 2) / 2, self.dt]])  # Process error (covariance)
        # budget model
        self.A_budget = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 1]])
        self.x_budget = np.array([0, 0, 0])
        self.Q_budget = 0.5 * np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]) 
        self.bridge_priority = np.random.uniform(0.1,0.3) # bridge priority 

        # interventions true state    
        self.int_true_var = np.array([[10**-8, 0.05**2, 10**-10],
                                [2**2, 0.1**2, 10**-8],
                                [4**2, 0.15**2, 10**-8]])
        self.int_true = np.array([[0.5, 0.2, 1e-2],
                                [7.5, 0.3, 1e-2],
                                [18.75, 0.4, 1e-2]])
        # true initial state    
        self.x_init = np.array([[np.random.randint(35,90), np.random.uniform(-1.5,-0.15), -0.005],
                                [10**2, 0.05**2, 0.001**2]])
        # true states
        self.cs = np.empty((1,6))
        self.c_cs = np.empty((self.num_b, np.max(self.num_c), 3))
        self.b_cs = np.empty((self.num_b, 3))
        self.net_cs = np.empty((1,3))

        # transformation function 
        self.n_tr = 4
        self.ST = SpaceTransformation(self.n_tr,self.max_cond,self.min_cond)
        self.SC = StateConstraints()
        self.ME = MixtureEstimate()

        # estimated states
        self.e_Ex = np.empty((1,6))
        self.e_Var = np.empty((1,6,6))
        self.c_Ex = np.empty((self.num_b,np.max(self.num_c),3))
        self.c_Var = np.empty((self.num_b,np.max(self.num_c),3,3))
        self.b_Ex = np.empty((self.num_b,3))
        self.b_Var = np.empty((self.num_b,3,3))
        self.net_Ex = np.empty((1,6))
        self.net_Var = np.empty((1,6,6))

        # actions tracking
        self.act_timer = 0      # tracking the frequency of the expert's actions
        self.act_type_1 = 0     # tracking action type 1
        self.act_type_2 = 0     # tracking action type 2
        self.act_type_3 = 0     # tracking action type 3
        self.act_type_4 = 0     # tracking action type 4
        
        # initilizing the network
        self.inspector = np.nan * np.zeros_like(self.num_e[self.cb, self.cc, 0])
        self.current_year = 0
            
        # initilizing the deterioration model
        self.Am = sp.block_diag(self.A, np.eye(3))
        self.Qm = sp.block_diag(self.Q, np.zeros(3))
        self.Fm = np.append(self.F, np.zeros([1,3]))

        # SSM model estimated parameters
        self.sigma_v = np.random.uniform(0.85,1.1)*self.inspector_std
        self.int_Ex = np.array([[[0.5,0.169,1e-2],[8.023,0.189,1e-2],[18.117,0.179,1e-2]]])
        self.int_var = np.array([[[8.28e-05,2.65e-07,-5.29e-08],[2.65e-07,0.00030,1.75e-05],[-5.29e-08,1.75e-05,0.00012]],
                            [[0.37,0.00087,-8.52e-05],[0.00087,0.0048,2.62e-05],[-8.52e-05,2.62e-05,0.00013]],
                            [[0.18,0.00027,-0.0017],[0.00027,0.00029,1.45e-05],[-0.0018,1.45e-05,0.00052]]])
        self.int_Q = np.square(np.array([[[1.39,0,0,0,0,0],[0,0.1,0,0,0,0],[0,0,0.045,0,0,0],[0,0,0,1.39,0,0],[0,0,0,0,0.1,0],[0,0,0,0,0,0.045]],
                        [[3.533,0,0,0,0,0],[0,0.747,0,0,0,0],[0,0,0.047,0,0,0],[0,0,0,3.533,0,0],[0,0,0,0,0.747,0],[0,0,0,0,0,0.047]],
                        [[3.768,0,0,0,0,0],[0,0.227,0,0,0,0],[0,0,0.0499,0,0,0],[0,0,0,3.768,0,0],[0,0,0,0,0.227,0],[0,0,0,0,0,0.0499]]]))
        self.init_var = np.array([3**2, 0.05**2, 0.005**2, 1, 1, 1])
        data_dir ='./data'
        filename = pjoin(data_dir, 'service_life.mat')
        self.CDF = sio.loadmat(filename)

        # RL model
        # actions on elements
        self.actions = np.array([0,1,2,3,4])
        self.actionCardinality = self.actions.shape[0]

        # actions on category
        self.action_h_c = np.array([0, 1])
        self.actionCardinality_h_c = self.action_h_c.shape[0]

        # actions on bridge
        self.action_h_b = np.array([0, 1])
        self.actionCardinality_h_b = self.action_h_b.shape[0]

        # solve an interval problem
        self.find_interval = 1

        # metrics related to the agent's actions
        self.actions_count = np.zeros((1,self.actionCardinality))
        self.action_hist = np.zeros((self.actionCardinality,self.num_e.sum()))
        self.cat_act_ratio = np.zeros((self.actionCardinality,np.max(self.num_c)))
        self.act_timer_on = 1

        # Rewards
        self.shutdown_cond = 35
        self.shutdown_speed = -2
        self.functional_cond = 70
        self.element_critical_cond = 30
        self.element_critical_speed = -2.5
        self.s_goal = 70
        self.action_costs = np.array([[0, -0.025, -0.075, -0.15, -0.40],[0, -0.025, -0.075, -0.15, -0.40]])
        self.fund_priority = np.random.rand(self.total_years+1)
        self.shutdown_cost = 1
        self.prev_shutdown = 0
        self.prev_action = 0
        self.shutdown_cost_perYear = -0.60
        self.rewards = 0
        self.elem_rewards = 0
        self.cat_rewards = 0
        self.bridge_rewards = 0
        self.penalty = 0
        self.penalty_cat = 0
        self.penalty_bridge = 0
        self.max_cost = 0
        # reduction factors
        self.alpha1 = 1
        self.alpha2 = 1

        # Performance Measures
        self.compatiblity = 0
        self.criticality = 0
        self.compatibility_cat = 0
        self.compatibility_bridge = 0
        self.compatibility_net = 0
        self.variablity = 0

        # plotting
        self.color_ex = 'bo'
        self.color_std = 'b'
        self.plt_y = []
        self.plt_true_c = []
        self.plt_true_s = []
        self.plt_Ex = []
        self.plt_var = []
        self.plt_Ex_dot = []
        self.plt_var_dot = []
        self.plt_R = []
        self.plt_t = []

        # initiation
        
        self.get_initial_state()
        self.initial_run()
        # analyses level: network, bridge, category, element
        action = 0
        if self.network_lvl:
            if self.discrete_actions:
                _, _, _, observation = self.network_action_discrete(action)
            else:
                _, _, _, observation = self.network_action(action)
        elif self.bridge_lvl:
            if self.discrete_actions:
                _, _, _, observation = self.bridge_action_discrete(action)
            else:
                _, _, _, observation = self.bridge_action(action)
        elif self.category_lvl:
            if self.discrete_actions:
                _, _, _, observation = self.cat_action_discrete(action)
            else:
                _, _, _, observation = self.cat_action(action)
        elif self.element_lvl:
            if self.discrete_actions:
                _, _, _, observation = self.elem_action_discrete(action)
            else:
                _, _, _, observation = self.elem_action(action)
        return observation
    """ Initial Environment Run ########################################################################################################

        get_initial_state : generate the initial state for structural elements
        initial_run : Propagate the initial state from t = 0, to t = 1 without an intervention

    """

    def get_initial_state(self):
        for i in range(self.num_b):
            for j in range(self.num_c[i]):
                for k in range(0,self.num_e[i,j,0]):
                    sample_state = np.random.multivariate_normal(self.x_init[0,:],np.diag(self.x_init[1,:]))
                    self.initial_state[i,j,k,:] = sample_state
        # initilize budget
        self.x_budget[0] = self.budget

    def initial_run(self):
        self.cs = self.initial_state
        self.e_Ex = np.concatenate([np.zeros(self.cs.shape), np.zeros(self.cs.shape)],axis = 3)
        self.e_Ex[:,:,:,0:3] = self.cs * np.random.uniform(0.85,1.05)
        self.e_Ex[self.e_Ex[:,:,:,1]>0, 1] = 0
        self.e_Ex[:,:,:,2] = 0
        e_var = np.diag(np.array(self.init_var))
        self.e_Var = np.zeros([self.num_b, np.max(self.num_c), self.num_e.max(),6,6])
        self.e_Var[:, :, :, :, :] = e_var
        for i in range(self.num_b):
            self.cb = i
            for j in range(self.num_c[i]):
                self.cc = j
                for k in range(self.num_e[i,j,0]):
                    self.ci = k
                    self.true_action(0)
                    if self.deterministic_model == 0:
                        self.estimate_step(self.e_Ex[self.cb,self.cc,self.ci,:], self.e_Var[self.cb, self.cc, self.ci, :, :], 0)
                self.category_state_update()
            self.bridge_state_update()
        self.network_state_update()
        self.cb = 0
        self.cc = 0
        self.ci = 0
        self.ec = self.num_e[self.cb,self.cc,0]

    def step(self, action): # action is a scalar
        # action 
        if self.network_lvl:
            if self.discrete_actions:
                _, _, reward, observation = self.network_action_discrete(action)
            else:
                _, _, reward, observation = self.network_action(action)
        elif self.bridge_lvl:
            if self.discrete_actions:
                _, _, reward, observation = self.bridge_action_discrete(action)
            else:
                _, _, reward, observation = self.bridge_action(action)
        elif self.category_lvl:
            if self.discrete_actions:
                _, _, reward, observation = self.cat_action_discrete(action)
            else:
                _, _, reward, observation = self.cat_action(action)
        elif self.element_lvl:
            if self.discrete_actions:
                _, _, reward, observation = self.elem_action_discrete(action)
            else:
                _, _, reward, observation = self.elem_action(action)
        # done
        done = 0
        # info
        info = {'e_compatibility': 1 - self.compatiblity/self.current_year, 'e_criticality': self.criticality/self.current_year,\
                'c_compatibility': 1 - self.compatibility_cat/self.current_year, 'b_compatibility': 1 - self.compatibility_bridge/self.current_year,\
                'n_compatibility': 1 - self.compatibility_net/self.current_year}
        return observation, reward, done, info
    """ Environment Actions ########################################################################################################

        network_action
        network_action_discrete
        bridge_action
        bridge_action_discrete
        cat_action
        cat_action_discrete
        elem_action
        elem_action_discrete
    """
    def network_action(self, goal):
        reward = 0
        # prepare bridge state
        state = self.state_net_prep(goal, 1)

        new_goal = copy.copy(goal)
        
        if self.deterministic_model:
            state_sum = np.sum(self.b_cs[:, 0 ])
        else:
            state_sum = np.sum(self.b_Ex[:, 0 ])

        for i in range(self.num_b):
            # state on goal
            goal_dim = 0

            # sort elements once
            if i == 0:
                if self.deterministic_model:
                    min_bridge_cond = np.argsort(self.b_cs[:, 0])
                else:
                    min_bridge_cond = np.argsort(self.b_Ex[:, 0])

            # element in category
            self.cb = min_bridge_cond[i]

            # get category goal from bridge goal
            goal_bridge = self.goal_network_to_bridge(new_goal, state_sum)

            self.bridge_action(goal_bridge)
            # 
            self.network_state_update()
            # collect rewards
            self.get_rewards_bridge(state, goal)
            reward += self.bridge_rewards
            # prepare states to check on if the goal is acheived
            state_one_bridge_update = self.state_prep_3()
            # transition the goal
            new_goal = self.goal_transition(state, goal, state_one_bridge_update, goal_dim)
        # after step
        # budget transition
        fund_pr = self.fund_priority[self.current_year]
        self.update_budget(fund_pr, reward)
        self.current_year += 1
        self.act_timer_on = 1
        self.shutdown_cost = 1
        
        next_state = self.state_net_prep(goal, 0)

        # update plotting functions
        if self.plotting == 1:
            self.plot_log()

        return state, goal, reward, next_state

    def network_action_discrete(self, action):
        if action == 0:
            state, goal, reward, next_state = self.network_action(0)
        else:
            if self.deterministic_model:
                state_ = self.transform_to_original(self.net_cs[0:2])
            else:
                state_ = self.transform_to_original(self.net_Ex[0:2])
            if self.expert_model:
                if self.include_budget:
                    goal = self.expert_demonstrator(state_, self.x_budget[0])
                else:
                    goal = self.expert_demonstrator(state_, self.max_budget)
            else:
                goal = self.expert_demonstrator_noInterval(state_)
            state, goal, reward, next_state = self.network_action(goal)
        return state, goal, reward, next_state

    def bridge_action(self, goal):
        reward = 0
        # prepare bridge state
        state = self.state_bridge_prep(goal, 1)

        new_goal = copy.copy(goal)
        
        if self.deterministic_model:
            state_sum = np.sum(self.c_cs[self.cb, :, 0 ])
        else:
            state_sum = np.sum(self.c_Ex[self.cb, :, 0 ])

        for i in range(self.num_c[self.cb]):

            # state on goal
            goal_dim = 0

            # sort elements once
            if i == 0:
                if self.deterministic_model:
                    min_cat_cond = np.argsort(self.c_cs[self.cb, :, 0])
                else:
                    min_cat_cond = np.argsort(self.c_Ex[self.cb, :, 0])

            # element in category
            self.cc = min_cat_cond[i]

            # reset the element tracker for each category 
            self.ec = self.num_e[self.cb,self.cc,0]

            # get category goal from bridge goal
            goal_cat = self.goal_bridge_to_category(new_goal, state_sum)

            self.cat_action(goal_cat)
            # 
            self.bridge_state_update()
            # collect rewards
            self.get_rewards_cat(state, goal)
            reward += self.cat_rewards
            # prepare states to check on if the goal is acheived
            state_one_cat_update = self.state_prep_2()
            # transition the goal
            new_goal = self.goal_transition(state, goal, state_one_cat_update, goal_dim)
        # after step
        # budget transition
        if self.bridge_lvl:
            fund_pr = self.fund_priority[self.current_year]
            self.update_budget(fund_pr, reward)
            self.current_year += 1
            self.act_timer_on = 1
            self.shutdown_cost = 1
        
        next_state = self.state_bridge_prep(goal, 0)

        # update plotting functions
        if self.plotting == 1:
            self.plot_log()
        if self.bridge_lvl:
            return state, goal, reward, next_state

    def bridge_action_discrete(self, action):
        if action == 0:
            state, goal, reward, next_state = self.bridge_action(0)
        else:
            if self.deterministic_model:
                state_ = self.transform_to_original(self.b_cs[self.cb, 0:2])
            else:
                state_ = self.transform_to_original(self.b_Ex[self.cb, 0:2])
            if self.expert_model:
                if self.include_budget:
                    goal = self.expert_demonstrator(state_, self.x_budget[0])
                else:
                    goal = self.expert_demonstrator(state_, self.max_budget)
            else:
                goal = self.expert_demonstrator_noInterval(state_)
            state, goal, reward, next_state = self.bridge_action(goal)
        return state, goal, reward, next_state

    def cat_action(self, goal):
        reward = 0
        # prepare category state
        state = self.state_category_prep(goal, 1)

        new_goal = copy.copy(goal)

        for i in range(self.num_e[self.cb, self.cc, 0]):
            # state on goal
            goal_dim = 0
            # get action from goal
            action = self.expert_demonstrator_action(new_goal)
            # sort elements once
            if i == 0:
                if self.deterministic_model:
                    min_elem_cond = np.argsort(self.cs[self.cb, self.cc, 0:self.num_e[self.cb,self.cc, 0], 0])
                else:
                    min_elem_cond = np.argsort(self.e_Ex[self.cb, self.cc, 0:self.num_e[self.cb,self.cc, 0], 0])
            # element in category
            self.ci = min_elem_cond[i]
            # apply action
            self.elem_action(action)
            # state after action
            # get updated state of the category
            self.category_state_update()
            # collect rewards
            self.get_rewards_cat(state, goal)
            reward += self.cat_rewards
            # prepare states
            state_one_elem_update = self.state_prep_1()
            # transition the goal
            new_goal = self.goal_transition(state, goal, state_one_elem_update, goal_dim)

        # after step
        # budget transition
        if self.category_lvl:
            fund_pr = self.fund_priority[self.current_year]
            self.update_budget(fund_pr, reward)
            self.current_year += 1
            self.act_timer_on = 1
            self.shutdown_cost = 1
        
        next_state = self.state_category_prep(goal, 0)

        # update plotting functions
        if self.plotting == 1:
            self.plot_log()
        if self.category_lvl:
            return state, goal, reward, next_state
        
    def cat_action_discrete(self, action):
        if action == 0:
            state, goal, reward, next_state = self.cat_action(0)
        else:
            if self.deterministic_model:
                state_ = self.transform_to_original(self.c_cs[self.cb, self.cc, 0:2])
            else:
                state_ = self.transform_to_original(self.c_Ex[self.cb, self.cc, 0:2])
            if self.expert_model:
                if self.include_budget:
                    goal = self.expert_demonstrator(state_, self.x_budget[0])
                else:
                    goal = self.expert_demonstrator(state_, self.max_budget)
            else:
                goal = self.expert_demonstrator_noInterval(state_)
            state, goal, reward, next_state = self.cat_action(goal)
        return state, goal, reward, next_state

    def elem_action(self, action):
        reward = 0
        # prepare category state
        if self.element_lvl:
            state = self.state_element_prep()
        # check element compatibility
        self.compatibility_action(action, 0)
        # perform element action
        self.true_action(action)
        if self.deterministic_model == 0:
            self.estimate_step(self.e_Ex[self.cb,self.cc,self.ci,:], self.e_Var[self.cb, self.cc, self.ci, :, :], action)
        self.ec = self.ec - 1
        # update action count
        self.count_action(action)
        # check element compatibility
        self.compatibility_action(action, 1)
        # advance time
        if self.element_lvl:
            fund_pr = self.fund_priority[self.current_year]
            self.update_budget(fund_pr, reward)
            self.current_year += 1
            self.act_timer_on = 1
            self.shutdown_cost = 1
        # rewards 
        self.get_rewards_elem(action)

        reward = self.elem_rewards
        if self.element_lvl:
            next_state = self.state_element_prep() 
            return state, action, reward, next_state

    def elem_action_discrete(self, action):
        if action == 0: # no intervention
            state, action, reward, next_state = self.elem_action(0)
        else:
            if self.deterministic_model:
                state_ = self.transform_to_original(self.cs[self.cb, self.cc, self.ci, 0:2])
            else:
                state_ = self.transform_to_original(self.e_Ex[self.cb, self.cc, self.ci, 0:2])
            if self.expert_model:
                if self.include_budget:
                    goal = self.expert_demonstrator_noInterval(state_)
                    action = self.expert_demonstrator_action(goal)
                else:
                    goal = self.expert_demonstrator_noInterval(state_)
                    action = self.expert_demonstrator_action(goal)
            else:
                goal = self.expert_demonstrator_noInterval(state_)
                action = self.expert_demonstrator_action(goal)
            state, action, reward, next_state = self.elem_action(action)
        return state, action, reward, next_state
    """ - Rewards Section  ########################################################################################################

    get_rewards_elem : get the costs/rewards from actions on elements
    get_rewards_cat : get the costs/rewards from actions on categories
    get_rewards_bridge : get the costs/rewards from actions on bridges
    get_rewards_sparse : obtain the final reward wind/lose
    check_action_limits : determine if a penalty exist due to repeating the same action
    """
    def get_rewards_elem(self,action):
        self.penalty = 0
        sc_act = 0
        act_check = self.check_action_limits()
        # shutdown cost
        if self.shutdown_cost:
            sc_act =  self.action_costs[self.cc][action]
            self.shutdown_cost = 0
            self.prev_action = action
        else:
            if self.prev_action >= action:
                sc_act = 0
            else:
                sc_act = self.action_costs[self.cc][action] - self.action_costs[self.cc][self.prev_action]
                self.prev_action = action

        if action>0 :
            elem_cost = self.action_costs[self.cc][action] + sc_act
        else:
            elem_cost = 0

        if self.deterministic_model:
            if (self.cs[self.cb, self.cc, self.ci,0] < self.element_critical_cond or \
                self.cs[self.cb, self.cc, self.ci,1] < self.element_critical_speed or \
                (self.x_budget[0]<0 and action>0 and self.include_budget) or \
                act_check or self.prev_shutdown) :
                    self.penalty = 1    
        else:
            if (self.e_Ex[self.cb, self.cc, self.ci,0]<self.element_critical_cond or \
                self.e_Ex[self.cb, self.cc, self.ci,1]< self.element_critical_speed or \
                (self.x_budget[0]<0 and action>0 and self.include_budget) or \
                act_check or self.prev_shutdown) :
                    self.penalty = 1

        if self.penalty:
            elem_cost = elem_cost + self.action_costs[self.cc][3]
            self.penalty = 0

        self.elem_rewards = elem_cost

    def get_rewards_cat(self, state, goal):
        if self.penalty_cat != 1:
            if (state[0]<=self.shutdown_cond or state[1]<=self.shutdown_speed):
                self.penalty_cat = 1
        # check penalty
        if self.penalty_cat:
            cat_cost = self.action_costs[self.cc][3]
            self.penalty_cat = 0
        else:
            cat_cost = 0

        if goal < 0: goal = 0
        self.cat_rewards = - goal + self.elem_rewards + cat_cost      
    
    def get_rewards_bridge(self, state, goal):
        if self.penalty_bridge ==0:
            if (state[0]<=self.shutdown_cond or state[1]<-self.shutdown_speed):
                self.penalty_bridge = 1
        # check penalty
        if self.penalty_bridge:
            bridge_cost = self.action_costs[self.cc][3]
            self.penalty_bridge = 0
        else:
            bridge_cost = 0

        if goal < 0: goal = 0
        self.bridge_rewards = - goal + self.cat_rewards + bridge_cost     
    
    def get_rewards_sparse(self, reward_agent, reward_expert):
        if reward_agent > reward_expert:
            sparse_reward = 1
        else:
            sparse_reward = -1
        return sparse_reward
    
    def check_action_limits(self):
        act_penalty = 0
        if self.network_lvl:
            if self.act_type_4 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * self.num_b * 2 or \
                self.act_type_3 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * self.num_b * 4 or \
                self.act_type_2 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * self.num_b * 8 or \
                self.act_type_1 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * self.num_b * 16:
                act_penalty = 1
        elif self.bridge_lvl:
            if self.act_type_4 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * 2 or \
                self.act_type_3 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * 4 or \
                self.act_type_2 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * 8 or \
                self.act_type_1 > self.num_e[self.cb, self.cc, 0] * self.num_c[self.cb] * 16:
                act_penalty = 1
        elif self.category_lvl:
            if self.act_type_4 > self.num_e[self.cb, self.cc, 0] * 2 or \
                self.act_type_3 > self.num_e[self.cb, self.cc, 0] * 4 or \
                self.act_type_2 > self.num_e[self.cb, self.cc, 0] * 8 or \
                self.act_type_1 > self.num_e[self.cb, self.cc, 0] * 16:
                act_penalty = 1
        elif self.element_lvl:
            if self.act_type_4 > 2 or \
                self.act_type_3 > 4 or \
                self.act_type_2 > 8 or \
                self.act_type_1 > 16:
                act_penalty = 1
        return act_penalty
    """ - State Propagation in Time ########################################################################################################

        true_action : applies an action on the true state (relies on true_step)
        true_step : advances the true state from (t) to (t+1)
        estimate_step : applies an action on the deterioration state and advance the state from (t) to (t+1)
        update_budget : changes the budget over time
        gen_observations : generates observations from the true state
        count_action : counter for time since the last action over time + actions counter
    """
    
    # true_action : applies an action on the true state (relies on true_step)
    def true_action(self, action):
        stall_counter = 0
        max_stall = 50
        # no action
        if action == 0: # do nothing
            # space transformation to know min value in the transformed space
            min_cond,_ = self.ST.original_to_transformed(self.min_cond)
            cond_1 = self.cs[self.cb, self.cc, 0:self.num_e[self.cb,self.cc, 0], 0] <= min_cond
            if self.num_e[self.cb,self.cc, 0] < self.num_e.max():
                check_1 = self.cs[self.cb, self.cc, :, 0] != 0
                cond_1 = np.concatenate((cond_1, check_1[self.num_e[self.cb,self.cc, 0]:]))
            if cond_1[self.ci]:
                self.cs[self.cb, self.cc, cond_1, 0]=min_cond
                next_state = self.cs[self.cb, self.cc, self.ci, :]
            else:
                next_state = self.true_step(self.cs[self.cb, self.cc, self.ci, :], action)
                if next_state.size==next_state.shape[0]:
                    next_state = next_state[np.newaxis,:]
                while np.any(next_state[:,1]>0) or any(next_state[:,2]>0.005):
                    stall_counter += 1
                    next_state = self.true_step(self.cs[self.cb, self.cc, self.ci, :], action)
                    if next_state.size==next_state.shape[0]:
                        next_state = next_state[np.newaxis,:]
                    if stall_counter > max_stall and np.any(next_state[:,1]>0):
                        next_state[0,1] = -next_state[0,1]
                    if stall_counter > max_stall and any(next_state[:,2]>0.005):
                        next_state[0,2] = -next_state[0,2]


        elif action<4: # maintaine
            next_state = self.true_step(self.cs[self.cb, self.cc, self.ci, :], action)
            max_cond,_ = self.ST.original_to_transformed(self.max_cond)
            # check the limits
            if next_state.size==next_state.shape[0]:
                next_state = next_state[np.newaxis,:]
            # check max condition
            if np.any(next_state[:,0]>0.99*max_cond):
                const_ind = np.where(next_state[:,0]>0.99*max_cond)
                if const_ind[0].shape[0] == 1:
                    next_state[const_ind[0][0],0] = 0.99*max_cond
                else:
                    for i in range(const_ind[0].shape[0]):
                        next_state[const_ind[0][i],0] = 0.99*max_cond
            # check speed
            if np.any(next_state[:,1]>0):
                const_ind = np.where(next_state[:,1]>0)
                if const_ind[0].shape[0] == 1:
                    next_state[const_ind[0][0],1] = self.cs[self.cb, self.cc, self.ci,1]
                else:
                    for i in range(const_ind[0].shape[0]):
                        next_state[const_ind[0][i],1] = self.cs[self.cb, self.cc, self.ci[const_ind[0][i]],1]
            # check acc.
            if np.any(next_state[:,2]>0.01):
                const_ind = np.where(next_state[:,2]>0.01)
                if const_ind[0].shape[0] == 1:
                    next_state[const_ind[0][0],2] = self.cs[self.cb, self.cc, self.ci, 2]
                else:
                    for i in range(const_ind[0].shape[0]):
                        next_state[const_ind[0][i],2] = self.cs[self.cb, self.cc, self.ci[const_ind[0][i]], 2]
            
        else: # replace
            # space transformation to know max value in the transformed space
            max_cond,_ = self.ST.original_to_transformed(self.max_cond)
            next_state = np.array([max_cond, -0.1, -0.001])
        if next_state.shape[0] == next_state.size:
            next_state = next_state.squeeze()
        self.cs[self.cb, self.cc, self.ci,:] = next_state
        if self.deterministic_model == 0:
            self.gen_observations()

    def true_step(self, state, action):
        if action == 0:
            int_noise = np.array(np.zeros(3))
        else:
            true_mu = self.int_true[action-1,:]
            true_Sigma = np.diag(self.int_true_var[action-1,:])
            # check for action frequency
            if self.action_hist[action,self.ci] != 0:
                ind_year = 15 * (self.current_year - self.action_hist[action,self.ci])
                if ind_year>60:
                    self.alpha1 = 1
                    self.alpha2 = 1
                else:
                    self.alpha1 = 1 #self.CDF['CDF'][0,action-1][0,int(ind_year)]
                    self.alpha2 = 1 #self.CDF['CDF'][0,action+2][0,int(ind_year)]
                true_mu[0],true_Sigma[0,0] = true_mu[0]*self.alpha1,true_Sigma[0,0]*self.alpha1**2
                true_mu[1],true_Sigma[1,1] = true_mu[1]*self.alpha2,true_Sigma[1,1]*self.alpha2**2
            int_noise = np.random.multivariate_normal(true_mu,true_Sigma)
        if state.size>3:
            process_noise = np.random.multivariate_normal(np.zeros(3), self.Q, state.shape[0])
            int_noise = np.repeat(int_noise[np.newaxis,:],state.shape[0],axis=0)
        else:
            process_noise = np.random.multivariate_normal(np.zeros(3), self.Q)
        next_state = np.transpose(self.A@state.squeeze().transpose()) +process_noise + int_noise
        return next_state

    def estimate_step(self, mu, var, action):
        if action == 0 or action == 4:
            int_mu = np.zeros([3])
            int_Sigma = np.zeros([3,3])
            Q_int = np.zeros([6,6])
            self.Am[0:3,3:6] = np.zeros([3,3])
        else:
            int_mu = self.int_Ex[0][action-1]
            int_Sigma = np.diag(self.int_var[0][action-1])
            Q_int = self.int_Q[action-1]
            self.Am[0:3,3:6] = np.eye(3)
        all_noise = sp.block_diag(self.Q,np.zeros([3,3])) + Q_int
        if mu.size == mu.squeeze().shape[0]:
            mu = mu[np.newaxis,:]
            var = var[np.newaxis,:,:]
            # check for actions timeline
            if action != 4:
                if self.action_hist[action,self.ci] == 0:
                    self.action_hist[action,self.ci] = self.current_year
                else:
                    int_mu[0],int_Sigma[0,0] = int_mu[0]*self.alpha1,int_Sigma[0,0]*self.alpha1**2
                    int_mu[1],int_Sigma[1,1] = int_mu[1]*self.alpha2,int_Sigma[1,1]*self.alpha2**2
                    self.action_hist[action,self.ci] = self.current_year
        else:
            all_noise = np.repeat(all_noise[np.newaxis,:,:],np.size(self.ci),axis=0)
        int_mu = int_mu[np.newaxis,:]
        mu[:,3:6] = int_mu
        var[:,3:6, 3:6] = int_Sigma[np.newaxis,:,:]
        if mu.size == mu.shape[0]:
            mu = mu.squeeze()
            var = var.squeeze()
        if action != 4:
            mu_pred = (self.Am @ mu.transpose()).transpose()
            var_pred = self.Am @ var @ self.Am.transpose() + all_noise
            if action != 0:
                if np.any(mu_pred[:,1] + 2 * np.sqrt(var_pred[:,1, 1]) > 0):
                    mu[:,4:6] = [[0, 0]]
                    var[:,4:6, 4:6] = [[[1e-8, 0],[0, 1e-8]]]
                    Q_int[1:3,1:3] = [[1e-8, 0],[0, 1e-8]]
                    Q_int[4:6,4:6] = [[1e-8, 0],[0, 1e-8]]
                    all_noise = sp.block_diag(self.Q,np.zeros([3,3])) + Q_int
                    mu_pred = (self.Am @ mu.transpose()).transpose()
                    var_pred = self.Am @ var @ self.Am.transpose() + all_noise
            else:
                if mu_pred[0,1] == -np.inf:
                   mu_pred[0,1] =  mu[0,1]
        else:
            # space transformation to know max value in the transformed space
            max_cond,_ = self.ST.original_to_transformed(self.max_cond)
            mu_pred = np.array([max_cond, -0.1, -0.001, 0, 0, 0])
            var_pred = np.diag([1, 0.025**2, 0.0025**2, 0, 0, 0])
        # update with observations
        if np.any(~np.isnan(self.y[self.cc, self.ci])):
            Ie = np.eye(6)
            Fm = self.Fm[np.newaxis,:]
            er = self.y[self.cc, self.ci] - (Fm @ mu_pred.transpose()).squeeze()
            var_xy = (Fm@var_pred@Fm.transpose()).squeeze()+ self.inspector_std[self.inspector,1]**2
            if mu.size > mu.squeeze().shape[0]:
                Ie = np.repeat(Ie[np.newaxis,:,:],np.size(self.ci),axis=0)
                var_xy = var_xy[:,np.newaxis,np.newaxis]
                er = er[:,np.newaxis]
            Kg = var_pred@Fm.transpose()/var_xy
            mu_pred = mu_pred + Kg.squeeze() * er
            var_pred = (Ie - Kg@Fm)@var_pred
        if (mu_pred.size == mu_pred.squeeze().shape[0]) and len(mu_pred.shape)<2:
            mu_pred = mu_pred[np.newaxis,:]
            var_pred = var_pred[np.newaxis,:,:]
        # check constraints
        if np.any(mu_pred[:,1] + 2 * np.sqrt(var_pred[:,1, 1]) > 0):
            const_ind = np.where(mu_pred[:,1] + 2 * np.sqrt(var_pred[:,1, 1]) > 0)
            if const_ind[0].shape[0] == 1:
                mu_out, var_out = self.SC.state_constraints(mu_pred[const_ind[0][0],:], var_pred[const_ind[0][0],:,:])
                if mu_out[1] != -np.inf:
                    mu_pred[const_ind[0][0],:] = mu_out
                    var_pred[const_ind[0][0],:,:] = var_out
                else:
                    mu_pred[const_ind[0][0],1] = -1e-2
                    mu_pred[const_ind[0][0],2] = -1e-4
            else:
                for i in range(const_ind[0].shape[0]):
                    mu_out, var_out = self.SC.state_constraints(mu_pred[const_ind[0][i],:], var_pred[const_ind[0][i],:,:])
                    mu_pred[const_ind[0][i],:] = mu_out
                    var_pred[const_ind[0][i],:,:] = var_out

        # check min possible condition
        min_cond,_ = self.ST.original_to_transformed(self.min_cond)
        if np.any(mu_pred[:,0] < min_cond):
            const_ind = np.where(mu_pred[:,0] < min_cond)
            if const_ind[0].shape[0] == 1:
                mu_pred[const_ind[0][0],0] = min_cond
            else:
                for i in range(const_ind[0].shape[0]):
                    mu_pred[const_ind[0][i],0] = min_cond

        # check max possible condition
        max_cond,_ = self.ST.original_to_transformed(self.max_cond)
        if np.any(mu_pred[:,0] > max_cond):
            const_ind = np.where(mu_pred[:,0] > max_cond)
            if const_ind[0].shape[0] == 1:
                mu_pred[const_ind[0][0],0] = max_cond
            else:
                for i in range(const_ind[0].shape[0]):
                    mu_pred[const_ind[0][i],0] = max_cond

        if mu_pred.size == mu_pred.squeeze().shape[0]:
            mu_pred = mu_pred.squeeze()
            var_pred = var_pred.squeeze()
            if mu_pred[1] == -np.inf:
                mu_pred[1] =  mu[1]
        # update element estimate
        self.e_Ex[self.cb, self.cc, self.ci,:] = mu_pred
        self.e_Var[self.cb, self.cc, self.ci,:,:] = var_pred

    def update_budget(self, fund_pr, cost):
        if fund_pr < self.bridge_priority:
            self.x_budget[2] = self.budget * self.bridge_priority
        self.x_budget[1] = copy.deepcopy(cost)
        self.x_budget = (self.A_budget @ np.transpose([self.x_budget])).squeeze()
    
    def gen_observations(self):
        if self.ec == self.num_e[self.cb,self.cc,0]:
            self.inspector = np.random.randint(1,self.inspector_std[:,0].max(),1)
            self.y = np.nan * np.zeros([self.num_c[self.cb], np.max(self.num_e[self.cb,:, 0])])
        if np.mod(self.current_year,self.inspection_frq[self.cb])==0 and self.current_year != 0 or self.current_year == 1:
            obs_noise = np.random.multivariate_normal(np.zeros(self.inspector.shape[0]),
            np.diag(self.inspector_std[self.inspector,1]**2))
            self.y[self.cc, self.ci] = self.cs[self.cb, self.cc, self.ci, 0] + obs_noise
        else:
            self.y[self.cc, self.ci] = np.nan

    def count_action(self, action):
        # check time since last action 
        if action > 0 and self.act_timer_on:
            self.act_timer = 0
            self.act_timer_on = 0
        elif self.act_timer_on:
            self.act_timer += 1
        # check frequency of action type
        if action > 3 :
            self.act_type_4 += 1
        elif action > 2 :
            self.act_type_3 += 1
        elif action > 1 :
            self.act_type_2 += 1
        elif action > 0 :
            self.act_type_1 += 1

    """ Deteriroation State Aggregation ########################################################################################################

        category_state_update : aggregate the deterioration states of structural elements
        bridge_state_update : aggregate the deterioration states of structural categories
        network_state_update : aggregate the deterioration states of bridges
    """ 
    def category_state_update(self):
        weigts = np.ones(self.num_e[self.cb, self.cc, 0])/self.num_e[self.cb, self.cc, 0]
        if self.deterministic_model == 0:
            Ex_c, Var_c = self.ME.gaussian_mixture(weigts, self.e_Ex[self.cb,self.cc,:,:], self.e_Var[self.cb,self.cc,:,:,:])
            self.c_Ex[self.cb, self.cc,:] = Ex_c[0:3]
            self.c_Var[self.cb, self.cc, :, :] = Var_c[0:3,0:3]
        else:
            Ex_c = np.mean(self.cs[self.cb, self.cc, :], axis= 0)
            self.c_cs[self.cb, self.cc,:] = Ex_c
    
    def bridge_state_update(self):
        weigts = np.ones(self.num_c[self.cb])/self.num_c[self.cb]
        if self.deterministic_model == 0:
            Ex_b, Var_b = self.ME.gaussian_mixture(weigts, self.c_Ex[self.cb, :,:], self.c_Var[self.cb, :,:,:])
            self.b_Ex[self.cb,:] = Ex_b
            self.b_Var[self.cb,:,:] = Var_b
        else:
            Ex_b = np.mean(self.c_cs[self.cb, :], axis= 0)
            self.b_cs[self.cb,:] = Ex_b
    
    def network_state_update(self):
        weigts = np.ones(np.size(self.num_b))/self.num_b
        if self.deterministic_model == 0:
            Ex_net, Var_net = self.ME.gaussian_mixture(weigts, self.b_Ex[:], self.b_Var[:])
            self.net_Ex = Ex_net
            self.net_Var = Var_net
        else:
            Ex_net = np.mean(self.b_cs[:], axis= 0)
            self.net_cs = Ex_net

    """ Miscellaneous

        transform_to_original
        assemble_state
        state_category_prep
        state_bridge_prep
        state_net_prep
        state_prep_0
        state_prep_1
        state_prep_2
        state_prep_3

    """  

    def transform_to_original(self,state):
        or_state = np.zeros(state.shape[0])
        or_state[0] = self.ST.transformed_to_original(state[0]).copy()
        or_state[1],_,_,_,_= copy.copy(self.ST.transformed_to_original_speed(state[0],state[1],np.ones(1)))
        return or_state
    
    def assemble_state(self, state):
        #state = np.append(state, self.act_timer/self.num_e[self.cc])
        state = np.append(state, self.variablity)
        #state = np.append(state, self.act_type_1/(100))
        #state = np.append(state, self.act_type_2/(100))
        #state = np.append(state, self.act_type_3/(100))
        #state = np.append(state, self.act_type_4/(100))
        #state = np.append(state, self.bridge_priority)
        #state = np.append(state, self.x_budget[0]/5e5)
        #state = np.append(state, self.ec)
        return state
    
    def state_element_prep(self):
        # state before action
        state_original = self.state_prep_0()
        # state before action
        state = self.assemble_state(state_original)
        return state

    def state_category_prep(self, goal, compatibility_check):
        if compatibility_check == 1:
            # check goal compatibility
            self.compatibility_goal_cat(goal)
        # state before action
        state_original = self.state_prep_1()
        # state before action
        state = self.assemble_state(state_original)
        return state

    def state_bridge_prep(self, goal, compatibility_check):
        if compatibility_check == 1:
            # check goal compatibility
            self.compatibility_goal_bridge(goal)
        # state before action
        state_original = self.state_prep_2()
        # state before action
        state = self.assemble_state(state_original)
        return state
    
    def state_net_prep(self, goal, compatibility_check):
        if compatibility_check == 1:
            # check goal compatibility
            self.compatibility_goal_network(goal)
        # state before action
        state_original = self.state_prep_3()
        # state before action
        state = self.assemble_state(state_original)
        return state

    def state_prep_0(self):
        if self.deterministic_model:
            state_tr = copy.deepcopy(self.cs[self.cb, self.cc, self.ci, 0:2])
            self.variablity = 0
        else:
            state_tr = copy.deepcopy(self.e_Ex[self.cb, self.cc, self.ci, 0:2])
            self.variablity = 0
        # back-transform the space
        state = self.transform_to_original(state_tr)
        return state

    def state_prep_1(self):
        if self.deterministic_model:
            state_tr = copy.deepcopy(self.c_cs[self.cb, self.cc,0:2])
            self.variablity = np.std(self.cs[self.cb, self.cc, : ,0])
        else:
            state_tr = copy.deepcopy(self.c_Ex[self.cb, self.cc, 0:2])
            self.variablity = np.std(self.e_Ex[self.cb, self.cc, : ,0])
        # back-transform the space
        state = self.transform_to_original(state_tr)
        return state

    def state_prep_2(self):
        if self.deterministic_model:
            state_tr = copy.deepcopy(self.b_cs[self.cb, 0:2])
            self.variablity = np.std(self.c_cs[self.cb, :, 0])
        else:
            state_tr = copy.deepcopy(self.b_Ex[self.cb, 0:2])
            self.variablity = np.std(self.c_Ex[self.cb, :, 0])
        # back-transform the space
        state = self.transform_to_original(state_tr)
        return state

    def state_prep_3(self):
        if self.deterministic_model:
            state_tr = copy.deepcopy(self.net_cs[0:2])
            self.variablity = np.std(self.b_cs[: ,0])
        else:
            state_tr = copy.deepcopy(self.net_Ex[0:2])
            self.variablity = np.std(self.b_Ex[: ,0])
        # back-transform the space
        state = self.transform_to_original(state_tr)
        return state

    """ Plotting

        plot_log
        state_plot
        pdf_plot_

    """  
    def plot_log(self):
        self.plt_Ex.append(self.c_Ex[self.cc,0])
        self.plt_var.append(self.c_Var[self.cc][0,0])
        self.plt_Ex_dot.append(self.c_Ex[self.cc,1])
        self.plt_var_dot.append(self.c_Var[self.cc][1,1])
        self.plt_y.append(np.mean(self.y[self.cc, self.ci]))
        self.plt_R.append(np.mean(self.inspector_std[self.inspector,1]**2)+np.var(self.y[self.cc, self.ci]))
        self.plt_true_c.append(np.mean(self.cs[self.cc][self.ci,0]))
        self.plt_true_s.append(np.mean(self.cs[self.cc][self.ci,1]))
        self.plt_t.append(self.current_year)
            # max length of the plot fot the time series stored 
        self.plt_Ex = self.plt_Ex[-20:]
        self.plt_var = self.plt_var[-20:]
        self.plt_Ex_dot = self.plt_Ex_dot[-20:]
        self.plt_var_dot = self.plt_var_dot[-20:]
        self.plt_y = self.plt_y[-20:]
        self.plt_R = self.plt_R[-20:]
        self.plt_t = self.plt_t[-20:]
        self.plt_true_c = self.plt_true_c[-20:]
        self.plt_true_s = self.plt_true_s[-20:]
        
    def state_plot(self, mu, var, y, R, years, plot_type, ax1, ax2, *args):
        if plot_type == 'condition':
            mu_cond_original = self.ST.transformed_to_original(mu).copy()
            y_original = copy.copy(self.ST.transformed_to_original(y))
            r_above = copy.copy(self.ST.transformed_to_original(y + 2 * np.sqrt(R))) - y_original
            r_under = y_original - copy.copy(self.ST.transformed_to_original(y - 2 * np.sqrt(R)))
            std = np.sqrt(var)
            std_original_1p = self.ST.transformed_to_original(mu + std).copy()
            std_original_1n = self.ST.transformed_to_original(mu - std).copy()
            std_original_2p = self.ST.transformed_to_original(mu + 2 * std).copy()
            std_original_2n = self.ST.transformed_to_original(mu - 2 * std).copy()
            ax1.plot(years, mu_cond_original)
            ax1.plot(years, y_original, self.color_ex)
            if years.size>1:
                ax1.errorbar(years, y_original, np.array([r_under, r_above]), linestyle='dotted')
            ax1.fill_between(years, std_original_1n, std_original_1p, color=self.color_std, alpha=.1)
            ax1.fill_between(years, std_original_2n, std_original_2p, color=self.color_std, alpha=.1)
            ax1.set(xlabel='Time (Year)', ylabel='Condition')
            ax1.set_ylim([25, 100])
        elif plot_type == 'speed':
            mu_cond = args[0]
            mu_dot, std_dot_1p, std_dot_1n, std_dot_2p, std_dot_2n = self.ST.transformed_to_original_speed(mu_cond,
                                                                                                           mu, var)
            ax2.plot(years, mu_dot)
            ax2.fill_between(years, std_dot_1n, std_dot_1p, color=self.color_std, alpha=.1)
            ax2.fill_between(years, std_dot_2n, std_dot_2p, color=self.color_std, alpha=.1)
            ax2.set(xlabel='Time (Year)', ylabel='Speed')
        plt.draw()
    
    def pdf_plot_(self, mu, var, action, plot_type, ax1, ax2):
        sigma_x = np.sqrt(var)
        x_val_before = np.linspace(mu[0] - 4 * sigma_x[0], mu[0] + 4 * sigma_x[0], 100)
        x_val_after = np.linspace(mu[1] - 4 * sigma_x[1], mu[1] + 4 * sigma_x[1], 100)
        if plot_type == 'condition':
            ax1.plot(x_val_before, stats.norm.pdf(x_val_before, mu[0], sigma_x[0]))
            ax1.plot(x_val_after, stats.norm.pdf(x_val_after, mu[1], sigma_x[1]))
            ax1.set(xlabel='Condition', ylabel=f'PDF (action:{action})')
        elif plot_type == 'speed':
            ax2.plot(x_val_before, stats.norm.pdf(x_val_before, mu[0], sigma_x[0]))
            ax2.plot(x_val_after, stats.norm.pdf(x_val_after, mu[1], sigma_x[1]))
            ax2.set(xlabel='speed', ylabel=f'PDF (action:{action})')
        plt.draw()

    """ Test Functions ########################################################################################################

        test_env
        
    """  
    def test_env(self, episodes_num):
        state = self.reset()
        self.expert_model = 1
        self.include_budget = 0
        years = np.arange(0,self.total_years)
        mu_vec = np.zeros([3,np.max(years)+1])
        var_vec = np.zeros([3,3,np.max(years)+1])
        reward_vec = np.nan * np.zeros(np.max(years)+1)
        goal_vec = np.nan * np.zeros(np.max(years)+1)
        obs = np.nan * np.zeros(np.max(years)+1)
        R = np.nan * np.zeros(np.max(years)+1)
        state_action = []
        state_noaction = []
        episode_reward = 0
        total_rewards = 0
        total_compt_ = 0
        total_crtic_ = 0
        compt_ = 0
        crtic_ = 0 
        ep = 0
        while ep < episodes_num:
            for k in range(self.total_years):
                prev_state = copy.copy(state[0])
                goal = self.expert_framework(state)
                state, reward, _, step_info = self.step(goal)
                episode_reward += reward
                reward_vec[k] = episode_reward
                goal_vec[k] = goal
                if self.element_lvl:
                    compt_ += step_info['e_compatibility']
                    crtic_ += step_info['e_criticality']
                elif self.category_lvl:
                    compt_ += step_info['c_compatibility']
                elif self.bridge_lvl:
                    compt_ += step_info['b_compatibility']
                else:
                    compt_ += step_info['n_compatibility']
                if episodes_num == 1:
                    if self.deterministic_model:
                        if self.element_lvl:
                            mu_vec[:, k] = self.cs[self.cb,self.cc, self.ci, 0:3]
                        elif self.category_lvl:
                            mu_vec[:, k] = self.c_cs[self.cb,self.cc,0:3]
                        elif self.bridge_lvl:
                            mu_vec[:, k] = self.b_cs[self.cb,0:3]
                        else:
                            mu_vec[:, k] = self.net_cs[0:3]
                        var_vec[:,:,k] = np.zeros((3, 3))
                        obs[k] = np.nan
                        R[k] = np.nan
                        if self.category_lvl:
                            if ~np.isnan(self.y[self.cc, self.ci]):
                                obs[k] =  np.mean(self.y[self.cc, :])
                                R[k] = np.mean(self.inspector_std[self.inspector[:],1]**2)
                        elif self.element_lvl:
                            if ~np.isnan(self.y[self.cc, self.ci]):
                                obs[k] =  np.mean(self.y[self.cc, self.ci])
                                R[k] = np.mean(self.inspector_std[self.inspector[:],1]**2)
                    else:
                        if self.element_lvl:
                            mu_vec[:, k] = self.e_Ex[self.cb,self.cc, self.ci,0:3]
                            var_vec[:,:,k] = self.e_Var[self.cb,self.cc, self.ci, 0:3,0:3]
                        elif self.category_lvl:
                            mu_vec[:, k] = self.c_Ex[self.cb,self.cc,0:3]
                            var_vec[:,:,k] = self.c_Var[self.cb,self.cc,0:3,0:3]
                        elif self.bridge_lvl:
                            mu_vec[:, k] = self.b_Ex[self.cb,0:3]
                            var_vec[:,:,k] = self.b_Var[self.cb,0:3,0:3]
                        else:
                            mu_vec[:, k] = self.net_Ex[0:3]
                            var_vec[:,:,k] = self.net_Var[0:3,0:3]
                        obs[k] = np.nan
                        R[k] = np.nan
                        if self.category_lvl:
                            if ~np.isnan(self.y[self.cc, self.ci]):
                                obs[k] =  np.mean(self.y[self.cc, :])
                                R[k] = np.mean(self.inspector_std[self.inspector[:],1]**2)
                        elif self.element_lvl:
                            if ~np.isnan(self.y[self.cc, self.ci]):
                                obs[k] =  self.y[self.cc, self.ci]
                                R[k] = self.inspector_std[self.inspector[0],1]**2
                else:
                    state_action.append(prev_state) if goal > 0 else state_noaction.append(prev_state)
            total_rewards += episode_reward
            total_compt_ += compt_/self.total_years
            ep += 1
            if episodes_num != 1:
                if self.element_lvl:
                    total_crtic_ += crtic_/self.total_years
                    print(f"Episode rewards: {episode_reward:.2f}; compatibility: {100*compt_/self.total_years:.0f}%; criticality: {100*crtic_/self.total_years:.0f}%")
                else:
                    print(f"Episode rewards: {episode_reward:.2f}; compatibility: {100*compt_/self.total_years:.0f}%")
            state = self.reset()
            self.expert_model = 1
            self.include_budget = 0
            episode_reward = 0
            compt_ = 0
            crtic_ = 0
        if self.element_lvl:
            print(f'> Total Rewards: {total_rewards:.1f}, Average Compatibility: {100*total_compt_/episodes_num:.1f}%, Average Criticality: {100*total_crtic_/episodes_num:.1f}%')
        else:
            print(f'> Total Rewards: {total_rewards:.1f}, Average Compatibility: {100*total_compt_/episodes_num:.1f}%')
        fig = plt.figure(1)
        fig.clf()
        if episodes_num == 1:
            (ax1, ax2, ax3, ax4) = fig.subplots(4,1)
            self.state_plot(mu_vec[0,:],var_vec[0,0,:],obs,R,years,'condition',ax1, ax2)
            self.state_plot(mu_vec[1,:],var_vec[1,1,:],obs,R,years,'speed',ax1, ax2,mu_vec[0,:])
            ax3.plot(years, reward_vec)
            ax3.set(xlabel='Time (Year)', ylabel='Cumulative Cost')
            ax4.plot(years, goal_vec)
            ax4.set(xlabel='Time (Year)', ylabel='Goals')
        else:
            (ax1, ax2) = fig.subplots(2,1)
            ax1.hist(state_action, density=True)
            ax1.set(xlabel='Condition with action', ylabel='Probablity')
            ax2.hist(state_noaction, density=True)
            ax2.set(xlabel='Condition without action', ylabel='Probablity')
        fig.tight_layout()

    """ Performance Metrics  ########################################################################################################

        expert_judge
        compatibility_action
        compatibility_goal_cat
        compatibility_goal_bridge
        compatibility_goal_network

    """  

    def expert_judge(self, next_state):
        if next_state[0]>90:
            system_state = 'excellent'
        elif next_state[0] <= 90 and next_state[0] > 70:
            system_state = 'good'
        elif next_state[0] <= 70 and next_state[0] > self.shutdown_cond:
            system_state = 'not good'
        else:
            system_state = 'poor'
        return system_state
    
    def compatibility_action(self, action, action_flag):
        # before action is applied
        if action_flag == 0:
            if self.deterministic_model == 0:
                # check action compatibility
                if np.any(self.e_Ex[self.cb, self.cc, self.ci,0] > 50) and action == 4:
                    self.compatiblity += 1
                elif np.any(self.e_Ex[self.cb, self.cc, self.ci,0] > 80) and action > 2:
                    self.compatiblity += 1
                elif np.any(self.e_Ex[self.cb, self.cc, self.ci,0] < 30) and action < 3:
                    self.compatiblity += 1
                elif np.any(self.e_Ex[self.cb, self.cc, self.ci,1] > -0.1) and action == 1:
                    self.compatiblity += 1
            else:
                # check action compatibility
                if np.any(self.cs[self.cb, self.cc, self.ci,0] > 50) and action == 4:
                    self.compatiblity += 1
                elif np.any(self.cs[self.cb, self.cc, self.ci,0] > 80) and action > 2:
                    self.compatiblity += 1
                elif np.any(self.cs[self.cb, self.cc, self.ci,0] < 30) and action < 3:
                    self.compatiblity += 1
                elif np.any(self.cs[self.cb, self.cc, self.ci,1] > -0.1) and action == 1:
                    self.compatiblity += 1
        else:
            # after action is applied
            # check condition crticality
            if self.deterministic_model == 0:
                if np.any(self.e_Ex[self.cb, self.cc, self.ci,0] < self.element_critical_cond):
                    self.criticality += 1
            else:
                if np.any(self.cs[self.cb, self.cc, self.ci,0] < self.element_critical_cond):
                    self.criticality += 1
    
    def compatibility_goal_cat(self, goal):
        # check goal compatiblity
        if self.deterministic_model == 0:
            if self.c_Ex[self.cb, self.cc, 0] > self.functional_cond and goal > 0.9 * (1 - self.functional_cond / self.max_cond) :
                self.compatibility_cat += 1
            elif np.any(self.c_Ex[self.cb, self.cc,0] < self.shutdown_cond) and goal <= 0:
                self.compatibility_cat += 1
        else:
            if self.c_cs[self.cb, self.cc, 0] > self.functional_cond and goal > 0.9 * (1 - self.functional_cond / self.max_cond) :
                self.compatibility_cat += 1
            elif np.any(self.c_cs[self.cb, self.cc,0] < self.shutdown_cond) and goal <= 0:
                self.compatibility_cat += 1
    
    def compatibility_goal_bridge(self, goal):
        # check goal compatiblity
        if self.deterministic_model == 0:
            if self.b_Ex[self.cb,0] > self.functional_cond and goal > 0.9 * (1 - self.functional_cond / self.max_cond) :
                self.compatibility_bridge += 1
            elif np.any(self.b_Ex[self.cc,0] < self.shutdown_cond) and goal <= 0:
                self.compatibility_bridge += 1
        else:
            if self.b_cs[self.cb,0] > self.functional_cond and goal > 0.9 * (1 - self.functional_cond / self.max_cond) :
                self.compatibility_bridge += 1
            elif np.any(self.b_cs[self.cc,0] < self.shutdown_cond) and goal <= 0:
                self.compatibility_bridge += 1
    
    def compatibility_goal_network(self, goal):
        # check goal compatiblity
        if self.deterministic_model == 0:
            if self.net_Ex[0] > self.functional_cond and goal > 0.9 * (1 - self.functional_cond / self.max_cond) :
                self.compatibility_net += 1
            elif np.any(self.net_Ex[0] < self.shutdown_cond) and goal <= 0:
                self.compatibility_net += 1
        else:
            if self.net_cs[0] > self.functional_cond and goal > 0.9 * (1 - self.functional_cond / self.max_cond) :
                self.compatibility_net += 1
            elif np.any(self.net_cs[0] < self.shutdown_cond) and goal <= 0:
                self.compatibility_net += 1
    
    """ Fixed Decision maker  ########################################################################################################
        expert_framework
        expert_demonstrator
        expert_demonstrator_noInterval
        expert_demonstrator_action
        goal_bridge_to_category
        goal_network_to_bridge
        goal_to_action
        action_to_goal
        goal_transition

    """  
    def expert_framework(self, state):
        if self.discrete_actions:
            goal = 1 if (self.act_timer % self.exp_interval == 0 and self.act_timer != 0) else 0
            if self.element_lvl == 0:
                if state[0] > 70 and self.variablity < 10 and state[1] > -1.5:
                    goal = 0
            elif state[0] > 70 and state[1] > -1.5 and self.element_lvl:
                goal = 0
            if self.include_budget:
                if self.x_budget[0] < 0: 
                    goal = 0
        else:
            if self.include_budget:
                goal = self.expert_demonstrator(state, self.x_budget[0])
                if self.element_lvl:
                    goal = self.expert_demonstrator_action(goal)
            else:
                goal = self.expert_demonstrator(state, self.max_budget)
                if self.element_lvl:
                    goal = self.expert_demonstrator_action(goal)
        return goal

    def expert_demonstrator(self, state, budget):
        if state[0] <= self.shutdown_cond:
            goal = (100-state[0])/75
        elif state[0] < 55 and self.act_timer % self.exp_interval == 0 and self.act_timer != 0 and budget > 0:
            goal = self.int_Ex[0][2,0]/75
        elif state[0] < 70 and self.act_timer % self.exp_interval == 0 and self.act_timer != 0 and budget > 0:
            goal = self.int_Ex[0][1,0]/75
        elif self.variablity > 10 and state[0]<60 and self.act_timer % self.exp_interval == 0 and self.act_timer != 0 and budget > 0:
            goal = self.int_Ex[0][2,0]/(75*2)
        elif state[1] < -1.5 and self.act_timer % self.exp_interval == 0 and self.act_timer != 0 and budget > 0:
            goal = self.int_Ex[0][0,0]/75
        else:
            goal = 0
        return goal

    def expert_demonstrator_noInterval(self, state):
        if state[0] <= self.shutdown_cond:
            goal = (100-state[0])/75
        elif state[0] < 55 and self.act_timer != 0:
            goal = self.int_Ex[0][2,0]/75
        elif state[0] < 70  and self.act_timer != 0:
            goal = self.int_Ex[0][1,0]/75
        elif self.variablity > 10 and state[0]<60:
            goal = self.int_Ex[0][2,0]/(75*2)
        else:
            goal = self.int_Ex[0][0,0]/75
        return goal
    
    def expert_demonstrator_action(self, goal):
        if goal > (self.int_Ex[0][2,0]/75):
            index_a = 4
        elif goal > (self.int_Ex[0][1,0]/75):
            index_a = 3
        elif goal > (self.int_Ex[0][0,0]/75):
            index_a = 2
        elif goal > 0:
            index_a = 1
        else:
            index_a = 0
        return index_a
    
    def goal_bridge_to_category(self, goal, state_sum):
        cat_goal = ( self.c_cs[self.cb,self.cc,0]/ state_sum) * goal * self.num_c[self.cb] if self.deterministic_model else ( self.c_Ex[self.cb,self.cc,0]/ state_sum ) * goal * self.num_c[self.cb]
        return cat_goal
    
    def goal_network_to_bridge(self, goal, state_sum):
        bridge_goal = ( self.b_cs[self.cb,0]/ state_sum) * goal * self.num_b if self.deterministic_model else ( self.b_Ex[self.cb,0]/ state_sum ) * goal * self.num_b
        return bridge_goal

    def goal_to_action(self, goal):
        action = 0 if (goal == 0) else 1
        return action

    def action_to_goal(self, action, state):
        goal = 0 if (action == 0) else self.expert_demonstrator_noInterval(state)
        return goal

    def goal_transition(self, state, goal, next_state, goal_dim):
        # return next goal
        # max (next_goal, 0) to avoid negative goals
        # min (next_goal, goal) to avoid actions due to deterioration
        return np.min([np.max([(state[goal_dim] + goal*75 - next_state[goal_dim])*self.num_e[self.cb,self.cc,0]/75, 0]), goal])

    """ gym requirments  ########################################################################################################
        render
        close
    """
    def render(self, mode = "human"):
        return
    
    def close(self):
        return

    """ Utilities ########################################################################################################
        ParamDictionary
        MixtureEstimate
        SpaceTransformation
        StateConstraints
    """
class ParamDictionary(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" %attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d

class MixtureEstimate:
    def __init__(self) -> None:
        pass

    def gaussian_mixture(self, weight, mu, var):
        mu_mix = np.empty(mu.shape)
        var_mix = np.empty(var.shape)
        mix_var = np.empty(var_mix.shape)
        for i in range(weight.size):
            mu_mix[i,:] = weight[i] * mu[i,:]
            var_mix[i,:, :] =  weight[i] * var[i,:, :]
        mu_mix = np.nansum(mu_mix,axis=0)
        var_mix = np.nansum(var_mix,axis=0)
        for i in range(weight.size):
            mix_var[i,:, :] = weight[i] * np.transpose([mu[i,:] - mu_mix])@ (mu[i,:] - mu_mix)[np.newaxis,:]
        mix_var = np.nansum(mix_var,axis=0)
        var_mix = var_mix + mix_var
        return mu_mix, var_mix

class SpaceTransformation:
    def __init__(self, n, upper_limit, lower_limit):
        self.n = 2 ** n
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.max_x = sc.gammaincinv(1 / self.n, 0.999) ** (1 / self.n)

    def original_to_transformed(self, y_original):
        y_tr = np.zeros(y_original.size)
        y_tr_s = np.zeros(y_original.size)
        if y_original.size == 1:
            y_tr, y_tr_s = self.compute_y_tr(y_original)
        else:
            for i in range(y_original.size):
                y_tr[i], y_tr_s[i] = self.compute_y_tr(y_original[i])
        return y_tr, y_tr_s

    def transformed_to_original(self, y_trans):
        y = np.zeros(y_trans.size)
        if y.shape[0] == 1:
            y = self.compute_y(y_trans)
        else:
            for i in range(y_trans.shape[0]):
                y[i] = self.compute_y(y_trans[i])
        return y
            

    def transformed_to_original_speed(self, mu, mu_dot, var_dot):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([self.lower_limit - max_range, self.upper_limit + max_range], [-self.max_x, self.max_x])
        std_dot = np.sqrt(var_dot)
        mu_dot_original = np.zeros(mu_dot.size)
        std_dot_1p = np.zeros(mu_dot.size)
        std_dot_1n = np.zeros(mu_dot.size)
        std_dot_2p = np.zeros(mu_dot.size)
        std_dot_2n = np.zeros(mu_dot.size)
        if mu.size == 1:
            if mu > self.upper_limit + max_range:
                mu = self.upper_limit + max_range
            if mu < self.lower_limit - max_range:
                mu = self.lower_limit - max_range
        else:
            if any(mu > self.upper_limit + max_range):
                ind_ = np.where(mu > self.upper_limit + max_range)
                mu[ind_] = self.upper_limit + max_range
            if any(mu < self.lower_limit - max_range):
                ind_ = np.where(mu < self.lower_limit - max_range)
                mu[ind_] = self.lower_limit - max_range
        if mu_dot.size==1:
            mu_s = fy_tr(mu)
            mu_dot_original = mu_dot * self.derivative_g(mu_s, self.n)
            std_dot_1p = (mu_dot + std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_1n = (mu_dot - std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_2p = (mu_dot + 2 * std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_2n = (mu_dot - 2 * std_dot) * self.derivative_g(mu_s, self.n)
        else:
            for i in range(mu_dot.size):
                if mu[i] < self.lower_limit - max_range:
                    mu[i] = self.lower_limit - max_range
                mu_s = fy_tr(mu[i])
                mu_dot_original[i] = mu_dot[i] * self.derivative_g(mu_s, self.n)
                std_dot_1p[i] = (mu_dot[i] + std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_1n[i] = (mu_dot[i] - std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_2p[i] = (mu_dot[i] + 2 * std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_2n[i] = (mu_dot[i] - 2 * std_dot[i]) * self.derivative_g(mu_s, self.n)
        return mu_dot_original, std_dot_1p, std_dot_1n, std_dot_2p, std_dot_2n

    @staticmethod
    def derivative_g(x, n):
        return n * np.exp(-x ** n) / sc.gamma(1 / n)

    def compute_y_tr(self, y_original):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([-self.max_x, self.max_x], [self.lower_limit - max_range, self.upper_limit + max_range])
        fy_s = interpolate.interp1d([self.lower_limit, self.upper_limit], [-1, 1])
        if y_original > self.upper_limit:
            y_s = 0.999
            y_tr_s = sc.gammaincinv(
                1 / self.n, np.abs(y_s)) ** (1 / self.n)
            y_tr = fy_tr(y_tr_s)
        elif y_original > (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            if y_original == self.upper_limit:
                y_original = self.upper_limit.copy() - 0.0001
            y_s = fy_s(y_original)
            if y_s > 0.999:
                y_s = 0.999
            y_tr_s = sc.gammaincinv(1 / self.n, np.abs(y_s)) ** (1 / self.n)
            if y_tr_s > self.max_x:
                y_tr_s = self.max_x.copy()
            y_tr = fy_tr(y_tr_s)
        elif y_original == (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            y_tr_s = 0
            y_tr = fy_tr(y_tr_s)
        elif y_original < (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            if y_original == self.lower_limit:
                y_original = self.lower_limit + 0.0001
            y_s = fy_s(y_original)
            y_tr_s = -sc.gammaincinv(1 / self.n, np.abs(y_s)) ** (1 / self.n)
            if y_tr_s < -self.max_x:
                y_tr_s = -self.max_x.copy()
            y_tr = fy_tr(y_tr_s)
        else:
            y_tr_s = np.NaN
            y_tr = np.NaN
        return y_tr, y_tr_s

    def compute_y(self, y_trans):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([self.lower_limit - max_range, self.upper_limit + max_range], [-self.max_x, self.max_x])
        fy_s = interpolate.interp1d([-1, 1], [self.lower_limit, self.upper_limit])
        y_s = np.zeros(y_trans.size)
        if y_trans > self.upper_limit + max_range:
            y_trans = self.upper_limit + max_range
        elif y_trans < self.lower_limit - max_range:
            y_trans = self.lower_limit - max_range
        y_s = fy_tr(y_trans)
        if y_s > self.max_x:
            y_s = self.max_x.copy()
            y_s_tr = sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s > 0:
            y_s_tr = sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s < -self.max_x:
            y_s = -self.max_x.copy()
            y_s_tr = -sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s == 0:
            y = (self.upper_limit - self.lower_limit)/2 + self.lower_limit
        elif y_s < 0:
            y_s_tr = -sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        else:
            y = np.NaN
        return y

class StateConstraints:
    def __init__(self):
        self.inv = np.linalg.pinv
        self.svd = np.linalg.svd
        self.D = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.d = np.array([-50, 0])
    
    def state_constraints(self, mu_kf, var_kf):
        mu_kf = mu_kf[:,np.newaxis]
        u_trunc, w_trunc, v_trunc = self.svd(var_kf)
        w_trunc = np.diag(w_trunc)
        amgs = np.sqrt(w_trunc) @ u_trunc.T @ np.transpose([self.D[0, :]])
        w, s = self.gram_schmidt_transformation(amgs)
        std_trunc = np.sqrt(self.D[1, :] @ var_kf @ np.transpose([self.D[1, :]]))
        s = (s * std_trunc) / w
        c_trunc = (self.d[0] - self.D[0, :] @ mu_kf) / std_trunc
        d_trunc = (self.d[1] - self.D[1, :] @ mu_kf) / std_trunc
        alpha = np.sqrt(2 / np.pi) / (mt.erf(d_trunc / np.sqrt(2)) - mt.erf(c_trunc / np.sqrt(2)))
        mu = alpha * (np.exp(-c_trunc ** 2 / 2) - np.exp(-d_trunc ** 2 / 2))
        sigma = alpha * (
                np.exp(-c_trunc ** 2 / 2) @ (c_trunc - 2 * mu) - np.exp(-d_trunc ** 2 / 2) @ (d_trunc - 2 * mu)) + mu ** 2 + 1
        mu_z = np.transpose([np.zeros(mu_kf.shape[0])])
        sigma_z = np.eye(var_kf.shape[0])
        mu_z[0, 0] = mu
        sigma_z[0, 0] = sigma
        mu_kf_new = mu_kf + u_trunc @ np.sqrt(w_trunc) @ s.T @ mu_z
        var_kf_new = u_trunc @ np.sqrt(w_trunc) @ s.T @ sigma_z @ s @ np.sqrt(w_trunc) @ u_trunc.T
        mu_kf_new = mu_kf_new.squeeze()
        return mu_kf_new, var_kf_new 

    @staticmethod
    def gram_schmidt_transformation(amgs):
        m, n = np.shape(amgs)
        w = np.zeros([n, n])
        t = np.zeros([m + n, m])
        n_range = range(n)
        for k in n_range:
            sigma = np.sqrt(amgs[:, k].T.dot(amgs[:, k]))
            if np.abs(sigma) < 100 * np.spacing(1):
                break
            w[k, k] = sigma
            for j in n_range[k + 1:]:
                w[k, j] = amgs[:, k].T.dot(amgs[:, j]) / sigma
            t[k, :] = amgs[:, k] / sigma
            for j in n_range[k + 1:]:
                amgs[:, j] = amgs[:, j] - w[k, j] * (amgs[:, k]) / sigma
        t[n:n + m, 0:m] = np.eye(m)
        index = n
        tot = range(n + m)
        for k in tot[n:]:
            temp = t[k, :]
            for i in range(k):
                temp = temp - t[k, :].dot(np.transpose([t[i, :]])).dot([t[i, :]])
            if np.linalg.norm(temp) > 100 * np.spacing(1):
                t[index, :] = temp / np.linalg.norm(temp)
                index = index + 1
        T = t[0:m, 0:m]
        return w, T