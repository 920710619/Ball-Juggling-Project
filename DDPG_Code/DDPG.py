import torch
import torch.nn.functional as F
import numpy as np
import model

BATCH_SIZE = 200
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

class DDPG:

    def __init__(self, state_dim, action_dim, action_lim, ram, device, Test):
        # Intialize parameter
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.action_lim  = action_lim
        self.count       = 0
        self.update      = 0
        self.ram         = ram
        self.device = device
        self.noise_size = 20

        # Initialze actor and critic network
        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.actor_pert   = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

        # Initialze target actor and critic network
        self.critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic = model.Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

        # Hard Update actor and critic network
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def get_action(self, state, Test=False, noise=True, param=True):
        if Test:
            noise = None
            param = None

        state = torch.from_numpy(state).to(self.device)
        self.actor.eval()
        self.actor_pert.eval()

        new_action = self.actor.forward(state).detach().data.cpu().numpy()
        self.actor.train()

        # Add Gaussion Noise to Explore
        if not Test:
            new_action += np.random.normal(loc=0.0, scale=self.noise_size, size = self.action_dim)
            new_action = np.clip(new_action, -1, 1)
       
        return new_action


    def optimize(self,Test):
        if Test:
            return

        self.count = self.count+1
        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
        
        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        r1 = torch.from_numpy(r1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)

        # ----- Optimize critic network ------

        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())

        y_expected = r1 + GAMMA*next_val
        y_expected = torch.squeeze(y_expected)

        y_predicted = torch.squeeze(self.critic.forward(s1, a1))

        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        # ------optimize actor ------
        pred_a1 = self.actor.forward(s1)
        # compute actor loss, and update the actor
        loss_actor = -1*torch.mean(self.critic.forward(s1, pred_a1))

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        # soft update target network
        self.soft_update(self.target_actor, self.actor, TAU)
        self.soft_update(self.target_critic, self.critic, TAU)
    
    def soft_update(self, target, source, tau):
        # Used to soft update the network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        # Used to hard update the network
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def save_best_models(self):
        # Save the actor network with best reward (best performance)
        torch.save(self.actor.state_dict(), './Models/best_actor.pt')
        print('Best model saved successfully')

    def save_cur_models(self):
        # Save the latest actor network
        torch.save(self.actor.state_dict(), './Models/cur_actor.pt')
        print('Current model saved successfully')

    def load_best_models(self):
        # Load the actor network with best reward (best performance)
        self.actor.load_state_dict(torch.load('./Models/best_actor.pt'))
        print('Best Models loaded succesfully')
        
    def load_cur_models(self):
        # Load the latest actor network
        self.actor.load_state_dict(torch.load('./Models/cur_actor.pt'))
        print('Current Models loaded succesfully')