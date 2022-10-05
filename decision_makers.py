
import numpy as np
import torch
import numpy as np
import os
from network import policy_net, dueling_policy_net

device = torch.device("cpu")

class decision_maker:
    def __init__(self, int_Ex, include_budget, shutdown_cond, discrete_actions, b_min_functional):
        self.int_Ex = int_Ex
        self.include_budget = include_budget
        self.exp_interval = 1 # np.random.randint(2,10,1)
        self.shutdown_cond = shutdown_cond
        self.discrete_actions = discrete_actions
        self.bridge_min_functional = b_min_functional
        self.filenames_bridge = ['params-hrl-infra_env-it(100000)-[28, 17].tar']#,'params-hrl-infra_env-it(100000)-[22, 15].tar'
        # Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        self.file_names = ['Beams-hrl-infra_env-it(100000)-[8, 23].tar','FrontWall-hrl-infra_env-it(100000)-[9, 21].tar', 
                            'Slabs-hrl-infra_env-it(100000)-[12, 15].tar','Gaurdrail-hrl-infra_env-it(100000)-[13, 14].tar', 
                            'WingWall-hrl-infra_env-it(100000)-[13, 21].tar','Pavement-hrl-infra_env-it(100000)-[19, 21].tar']
        if self.file_names != '':
            self.policy_nn_beams = policy_net(action_size=5, input_size=2).to(device)
            self.policy_nn_frontwall = policy_net(action_size=5, input_size=2).to(device)
            self.policy_nn_wingwall = policy_net(action_size=5, input_size=2).to(device)
            self.policy_nn_pavement = policy_net(action_size=5, input_size=2).to(device)
            self.policy_nn_slabs = policy_net(action_size=5, input_size=2).to(device)
            self.policy_nn_gaurdrail = policy_net(action_size=5, input_size=2).to(device)
            self.load_agent(self.file_names)
        
        if self.filenames_bridge != '':
            self.policy_bridge_1 = dueling_policy_net(action_size=2, input_size=3).to(device)
            self.load_agent_bridge(self.filenames_bridge)

    # element level decision makers
    def agent_element_lvl_discrete(self, state):#, val):
        if self.file_names != '':
            goal = self.evaluate_agent(state)
        else:
            goal = 1 #if (state[2] % self.exp_interval == 0) else 0 # and state[2] != 0
            if state[0] > 60 and state[1] > -1.2 :
                goal = 0
            if self.include_budget: 
                if state[-1] < 0: 
                    goal = 0
        return goal

    def agent_one_element_lvl(self, state, cc = 0):
        if self.discrete_actions:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif state[0] < 55 :
                goal = self.int_Ex[cc][2][0]/75
            elif state[0] < 70 :
                goal = self.int_Ex[cc][1][0]/75
            else:
                goal = self.int_Ex[cc][0][0]/75
        else:
            if self.file_names != '' :
                goal = self.evaluate_agent(state, cc)
            else:
                if state[0] <= self.shutdown_cond:
                    goal = (100-state[0])/75
                elif state[0] < 55 and state[2] % self.exp_interval == 0 and state[2] != 0 :
                    goal = self.int_Ex[cc][2][0]/75
                elif state[0] < 70 and state[2] % self.exp_interval == 0 and state[2] != 0 :
                    goal = self.int_Ex[cc][1][0]/75
                elif state[1] < -1.5 and state[2] % self.exp_interval == 0 and state[2] != 0:
                    goal = self.int_Ex[cc][0][0]/75
                else :
                    goal = 0
                # budget check
                if self.include_budget:
                    if state[-1] < 0 :
                        goal = 0
        return goal
    
    def agent_element_lvl(self, state, cc = 0):
        if self.file_names != '' :
            action = self.evaluate_agent(state, cc)
        else:
            if state[0] <= self.shutdown_cond:
                action = 4
            elif state[0] < 55 and state[2] % self.exp_interval == 0 and state[2] != 0 :
                action = 3
            elif state[0] < 70 and state[2] % self.exp_interval == 0 and state[2] != 0 :
                action = 2
            elif state[1] < -1.5 and state[2] % self.exp_interval == 0 and state[2] != 0:
                action = 1
            else :
                action = 0
                # budget check
            if self.include_budget:
                if state[-1] < 0 :
                    action = 0
        return action

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
        if self.file_names != '':
            goal = self.evaluate_agent_bridge(state)
        else:
            goal = 1 if (state[2] % 1 == 0 and state[2] != 0) else 0 #self.exp_interval
            if state[0] > 60 and state[1] > -1.5 and state[3] < 10 :
                goal = 0
            if self.include_budget: 
                if state[-1] < 0: 
                    goal = 0
        return goal

    def agent_high_lvl(self, state, cc):
        if self.discrete_actions:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif state[0] < 55 or state[3] > 15:
                goal = self.int_Ex[cc][2][0]/75
            elif state[0] < 70 or state[3] > 10:
                goal = self.int_Ex[cc][1][0]/75
            else:
                goal = self.int_Ex[cc][0][0]/75
        else:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif (state[0] < 55 or state[3] > 30) and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[cc][2][0]/75
            elif (state[0] < 70 or state[3] > 15) and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[cc][1][0]/75
            elif state[1] < -1.5 and state[2] % self.exp_interval == 0 and state[2] != 0:
                goal = self.int_Ex[cc][0][0]/75
            else :
                goal = 0
            # budget check
            if self.include_budget:
                if state[-1] < 0 :
                    goal = 0
        return goal

    def agent_high_lvl_bridge(self, state):
        if self.discrete_actions:
            if state[0] <= self.shutdown_cond:
                goal = 1
            elif state[0] < self.bridge_min_functional[0] or state[2] > self.bridge_min_functional[1]:
                goal = (self.bridge_min_functional[0] - state[0])/75
            else:
                goal = (100-state[0])/75
        else:
            if state[0] <= self.shutdown_cond:
                goal = (100-state[0])/75
            elif (state[0] < 55 or state[3] > 30) and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[0][2][0]/75
            elif (state[0] < 70 or state[3] > 15) and state[2] % self.exp_interval == 0 and state[2] != 0 :
                goal = self.int_Ex[0][1][0]/75
            elif state[1] < -1.5 and state[2] % self.exp_interval == 0 and state[2] != 0:
                goal = self.int_Ex[0][0][0]/75
            else :
                goal = 0
            # budget check
            if self.include_budget:
                if state[-1] < 0 :
                    goal = 0
        return goal
    
    def load_agent(self, filenames):
        file_path = []
        # load params file
        for i in range(len(filenames)):
            file_path.append(os.path.join(".", "model", filenames[i]))
        # initialize network weights
        # Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        self.policy_nn_beams.load_state_dict(torch.load(file_path[0])['policy_net'])
        self.policy_nn_frontwall.load_state_dict(torch.load(file_path[1])['policy_net'])
        self.policy_nn_slabs.load_state_dict(torch.load(file_path[2])['policy_net'])
        self.policy_nn_gaurdrail.load_state_dict(torch.load(file_path[3])['policy_net'])
        self.policy_nn_wingwall.load_state_dict(torch.load(file_path[4])['policy_net'])
        self.policy_nn_pavement.load_state_dict(torch.load(file_path[5])['policy_net'])
        return policy_net
    
    def load_agent_bridge(self, filenames):
        file_path = []
        # load params file
        for i in range(len(filenames)):
            file_path.append(os.path.join(".", "model", filenames[i]))
        # initialize network weights
        # Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        self.policy_bridge_1.load_state_dict(torch.load(file_path[0])['policy_net'])
        return None

    def evaluate_agent(self, state, cc):
        if cc == 0:
            action = torch.argmax(self.policy_nn_beams(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        elif cc == 1:
            action = torch.argmax(self.policy_nn_frontwall(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        elif cc == 2:
            action = torch.argmax(self.policy_nn_slabs(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        elif cc == 3:
            action = torch.argmax(self.policy_nn_gaurdrail(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        elif cc == 4:
            action = torch.argmax(self.policy_nn_wingwall(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        elif cc == 5:
            action = torch.argmax(self.policy_nn_pavement(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
            
        """ x = np.arange(25, 100, 0.1)
        y = np.arange(0, -3, -0.05)
        x_, y_ = np.meshgrid(x, y)
        action_map=np.zeros([y.size,x.size])
        for i in range(y.size):
            for j in range(x.size):
                action_map[i,j] = torch.argmax(self.policy_nn_beams(torch.FloatTensor([x_[i,j],y_[i,j]]).to(device)), dim = 1).cpu().detach().numpy() #(self.policy_nn(torch.FloatTensor([x_[i,j],y_[i,j]]).to(device))).cpu().detach().numpy()[0,1]/torch.max(self.policy_nn(torch.FloatTensor([x_[i,j],y_[i,j]]).to(device))).cpu().detach().numpy()
        import matplotlib.pyplot as plt 
        from mpl_toolkits.mplot3d import Axes3D
        plt.pcolor(x_,y_,action_map)
        plt.axvline(x=55)
        plt.axhline(y=-1.5)
        plt.xlabel('Condition $\mu_t$')
        plt.ylabel('Speed $\dot{\mu}_t$')
        #plt.title('Action $a_1$')
        plt.colorbar()
        plt.show() """
        return action[0]
    
    def evaluate_agent_bridge(self, state, cb = 0):
        if cb == 0:
            action = torch.argmax(self.policy_bridge_1(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        x = np.arange(25, 100, 0.1)
        y = np.arange(0, -3, -0.05)
        x_, y_ = np.meshgrid(x, y)
        action_map=np.zeros([y.size,x.size])
        for i in range(y.size):
            for j in range(x.size):
                action_map[i,j] = torch.argmax(self.policy_bridge_1(torch.FloatTensor([x_[i,j],y_[i,j], 20]).to(device)), dim = 1).cpu().detach().numpy() #(self.policy_nn(torch.FloatTensor([x_[i,j],y_[i,j]]).to(device))).cpu().detach().numpy()[0,1]/torch.max(self.policy_nn(torch.FloatTensor([x_[i,j],y_[i,j]]).to(device))).cpu().detach().numpy()
        import matplotlib.pyplot as plt 
        from mpl_toolkits.mplot3d import Axes3D
        plt.pcolor(x_,y_,action_map)
        plt.axvline(x=55)
        plt.axhline(y=-1.5)
        plt.xlabel('Condition $\mu_t$')
        plt.ylabel('Speed $\dot{\mu}_t$')
        #plt.title('Action $a_1$')
        plt.colorbar()
        plt.show()
        return action[0]

    def plot_policy(self, save_policy = 0):
        # load policy 

        #self.load_agent(self.file_names)

        #self.load_agent_bridge(self.filenames_bridge)
        import glob
        import pandas as pd

        glued_data = pd.DataFrame()
        for file_name in glob.glob('./Element_prediction/'+'*.csv'):
            x = pd.read_csv(file_name, low_memory=False,names=["Condition", "Speed"])
            glued_data = pd.concat([glued_data,x],axis=1)
        """ action = torch.argmax(self.policy_nn_frontwall(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        action = torch.argmax(self.policy_nn_slabs(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        action = torch.argmax(self.policy_nn_gaurdrail(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        action = torch.argmax(self.policy_nn_wingwall(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy()
        action = torch.argmax(self.policy_nn_pavement(torch.FloatTensor(state).to(device)), dim = 1).cpu().detach().numpy() """
        cost_curve = np.zeros(52)
        delta_q = np.zeros(52)
        j = 0
        for i in range(len(glob.glob('./Element_prediction/'+'*.csv'))):
            if i < 3:
                delta_q = np.cumsum(self.policy_nn_gaurdrail(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] - torch.max(self.policy_nn_gaurdrail(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy())
                cost_curve += (self.policy_nn_gaurdrail(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] + delta_q)/torch.max(self.policy_nn_gaurdrail(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy()
            elif i < 5:
                delta_q = np.cumsum(self.policy_nn_frontwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] - torch.max(self.policy_nn_frontwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy())
                cost_curve += (self.policy_nn_frontwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] + delta_q)/torch.max(self.policy_nn_frontwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy()
            elif i < 9:
                delta_q = np.cumsum(self.policy_nn_wingwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] - torch.max(self.policy_nn_wingwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy())
                cost_curve += (self.policy_nn_wingwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] + delta_q)/torch.max(self.policy_nn_wingwall(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy()
            elif i < 12:
                delta_q = np.cumsum(self.policy_nn_slabs(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] - torch.max(self.policy_nn_slabs(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy())
                cost_curve += (self.policy_nn_slabs(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] + delta_q)/torch.max(self.policy_nn_slabs(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy()
            elif i < 27:
                delta_q = np.cumsum(self.policy_nn_beams(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] - torch.max(self.policy_nn_beams(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy())
                cost_curve += (self.policy_nn_beams(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] + delta_q)/torch.max(self.policy_nn_beams(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy()
            else:
                delta_q = np.cumsum(self.policy_nn_pavement(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] - torch.max(self.policy_nn_pavement(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy())
                cost_curve += (self.policy_nn_pavement(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)).cpu().detach().numpy()[:,0] + delta_q)/torch.max(self.policy_nn_pavement(torch.FloatTensor(glued_data.iloc[:,[i+j,i+j+1]].to_numpy()).to(device)), dim =1 )[0].cpu().detach().numpy()
            j += 1
        import matplotlib.pyplot as plt 
        plt.plot(np.arange(2007, 2059, 1),cost_curve/29)
        plt.show()
        x = np.arange(25, 100, 0.1)
        y = np.arange(0, -3, -0.05)
        x_, y_ = np.meshgrid(x, y)
        action_map=np.zeros([y.size,x.size])
        for i in range(y.size):
            for j in range(x.size):
                action_map[i,j] = torch.argmax(self.policy_nn_pavement(torch.FloatTensor([x_[i,j],y_[i,j]]).to(device)), dim = 1).cpu().detach().numpy()
      
        import matplotlib.pyplot as plt 
        plt.rcParams.update({'font.size': 22})
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib
        if save_policy:
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))  # setup the plot
        plt.xlabel('Condition $\mu_t$')
        plt.ylabel('Speed $\dot{\mu}_t$')
        plt.title('Wing-wall')
        plt.plot([55, 100], [-1.5, -1.5], zorder = 2,label='Critical Speed', color = 'r',linewidth=2)
        plt.plot([55, 55], [-1.5, 0], zorder = 3,label='Critical Condition', color = 'r',linewidth=2)
        plt.plot([100, 100], [-2.99, -1.5], zorder = 3, color = 'r',linewidth=4)
        plt.plot([25, 55], [0, 0], zorder = 3, color = 'r',linewidth=4)
        plt.plot([25, 25], [-2.99, 0], zorder = 3, color = 'r',linewidth=4)
        plt.plot([25, 100], [-2.99, -2.99], zorder = 3, color = 'r',linewidth=4)
        plt.pcolor(x_,y_,action_map, zorder = 1)
        ax.set_xlim([25, 100])
        ax.set_ylim([-2.99, 0])
        if save_policy:
            plt.savefig('policy_map.pgf',bbox_inches='tight')
        else:
            plt.show()