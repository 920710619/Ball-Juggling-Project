import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Define Structure of Neural Network
        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs2 = nn.Linear(256,128)
        self.fca1 = nn.Linear(action_dim,128)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

        # Initialize Neural Network
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                torch.nn.init.uniform_(param, 0.1)


    def forward(self, state, action):
        # returns Value function Q(s,a) obtained from critic network

        s1 = swish(self.fcs1(state))
        s2 = swish(self.fcs2(s1))
        a1 = swish(self.fca1(action))
        x = torch.cat((s2,a1),dim=1)

        x = swish(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        # Define Structure of Neural Network
        self.fc1 = nn.Linear(state_dim,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,action_dim)

        # Initialize Neural Network
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                torch.nn.init.uniform_(param, 0.1)
                
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))

        return action



