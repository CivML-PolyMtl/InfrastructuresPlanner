import torch.nn as nn
import torch

class policy_net(nn.Module):
    def __init__(self, action_size, input_size=6):
        super(policy_net, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 64) #32#256
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 128)
        self.fc_adv = nn.Linear(64, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, self.action_size)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(x, torch.Tensor): x = torch.Tensor(x).to(device)
        if x.shape[0] == self.input_size:
            x = x.expand([1,self.input_size])
        x[:, 0] = x[:, 0] / 100
        #x[:, [0, 2, 4, 6, 8, 10]] = x[:, [0, 2, 4, 6, 8, 10]] / 100
        #x[:, 9] = x[:, 9] / 50
        #x[:, [2,3,4,5,6]] = x[:, [2,3,4,5,6]] / 50
        #x[:,[7,8,9,10]] = x[:,[7,8,9,10]] / 100
        #x[:,3] = x[:,3] / 10

        y = self.relu(self.fc1(x))
        adv = self.relu(self.fc_adv(y))
        Q = self.adv(adv) 
        return Q

class dueling_policy_net(nn.Module):
    def __init__(self, action_size, input_size=6):
        super(dueling_policy_net, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 64) #32#256
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 128)
        self.fc_adv = nn.Linear(64, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, self.action_size)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(x, torch.Tensor): x = torch.Tensor(x).to(device)
        if x.shape[0] == self.input_size:
            x = x.expand([1,self.input_size])
        x[:, 0] = x[:, 0] / 100
        #x[:, 2] = x[:, 2] / 100
        #x[:, [0, 2, 4, 6, 8, 10]] = x[:, [0, 2, 4, 6, 8, 10]] / 100
        #x[:, 9] = x[:, 9] / 50
        #x[:, [2,3,4,5,6]] = x[:, [2,3,4,5,6]] / 50
        #x[:,[7,8,9,10]] = x[:,[7,8,9,10]] / 100
        #x[:,3] = x[:,3] / 10

        y = self.relu(self.fc1(x))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = adv + value - advAverage
        return Q

