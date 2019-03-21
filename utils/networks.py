#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-03-19 12:16:41
@LastEditTime: 2019-03-20 22:20:26
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class RNNMLPNetwork_Critic_Adversary(nn.Module):
    def __init__(self, agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, shuffle=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(RNNMLPNetwork_Critic_Adversary, self).__init__()

        self.agent_id = agent_id
        self.n_agent_ori = n_agent_ori
        self.n_adversary_ori = n_adversary_ori
        self.n_agent = n_agent
        self.n_adversary = n_adversary
        self.n_neurons = input_dim
        self.shuffle = shuffle

        self.n_landmark = 2
        self.ag_obs_len = 2+2+2*self.n_landmark+(2+2)*(self.n_agent-1)+2*self.n_adversary
        self.ad_obs_len = 2+2+2*self.n_landmark+2*(self.n_adversary-1)+(2+2)*self.n_agent
        self.ag_act_len = 2
        self.ad_act_len=2

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        
        
        #self.rnn_agent = nn.RNN(2, 2*self.n_agent_ori) #o-a pair
        self.rnn_other_adversary = nn.RNN(self.ad_obs_len+self.ad_act_len,
                                 (self.ad_obs_len+self.ad_act_len)*(self.n_adversary_ori-1))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
        
    
    def init_hidden(self, n_neurons, batch_size, device):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, batch_size, n_neurons, device=device))
    
    def split(self, X):
        L = X.size(1)
        batch_size = X.size(0)
        # self is adversary
        n_landmark = self.n_landmark
        ag_obs_len = self.ag_obs_len
        ad_obs_len = self.ad_obs_len
        ag_act_len = self.ag_act_len
        ad_act_len = self.ad_act_len
        
        total_obs = ad_obs_len*self.n_adversary+ag_obs_len*self.n_agent
        X_obs, X_act = X[:, :total_obs], X[:,total_obs:]
        agent_obs = X_obs[:, self.n_adversary*ad_obs_len : self.n_adversary*ad_obs_len + self.n_agent*ag_obs_len] 
        agent_act = X_act[:, self.ag_act_len*self.n_adversary:self.ag_act_len*self.n_adversary+self.ag_act_len*self.n_agent]
        agent_X = torch.cat([agent_obs, agent_act], dim=1)
       
        other_adversary_obs = torch.cat([X_obs[:, : self.agent_id*ad_obs_len],
                                         X_obs[:, (self.agent_id + 1)*ad_obs_len : self.n_adversary*ad_obs_len]],
                                         dim=1)
        other_adversary_obs = other_adversary_obs.view([batch_size, self.n_adversary - 1, -1])
        other_adversary_act = torch.cat([X_act[:, : self.agent_id*ad_act_len],
                                         X_act[:, (self.agent_id + 1)*ad_act_len : self.n_adversary*ad_act_len]],
                                         dim=1)
        other_adversary_act = other_adversary_act.view([batch_size, self.n_adversary - 1, -1])
        other_adversary_X = torch.cat([other_adversary_obs,
                                         other_adversary_act],
                                         dim=2)
        self_X = torch.cat([X_obs[:,self.agent_id*ad_obs_len:(self.agent_id+1)*ad_obs_len],
                            X_act[:,self.agent_id*ad_act_len:(self.agent_id+1)*ad_act_len]],
                            dim=1)
        return agent_X, other_adversary_X, self_X
        
    def forward(self, X):# X[o0,o1,...,on, a0, a1,...,an]
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        batch_size = X.size(0)
    
        #print(f'x size:{X.size()}')
        if(self.shuffle):
            idx = torch.randperm(self.n_neurons)
            X = X[:, idx]
        X = self.in_fn(X)
        agent_X, other_adversary_X, self_X = self.split(X)
        #print(agent_X.size(), other_adversary_X.size(), self_X.size())
        # agent_X = agent_X.permute(1, 0, 2)
        other_adversary_X = other_adversary_X.permute(1,0,2)

        #print(f'x size after reshape:{X.size()}')
        #self.hidden_agent = self.init_hidden(2*self.n_agent_ori, batch_size, device=X.device)
        self.hidden_other_adversary = self.init_hidden((self.ad_obs_len+self.ad_act_len)*(self.n_adversary_ori-1), batch_size, device=X.device)
        #print(f'hidden size before rnn:{self.hidden.size()}') 
        #_, self.hidden_agent = self.rnn_agent(agent_X, self.hidden_agent)
        _, self.hidden_other_adversary = self.rnn_other_adversary(other_adversary_X, self.hidden_other_adversary)
        #print(f'hidden size after rnn:{self.hidden.size()}')  
        #self.hidden_agent = self.hidden_agent.view([batch_size, -1])
        self.hidden_other_adversary = self.hidden_other_adversary.view([batch_size,-1])
        X = torch.cat([agent_X, self.hidden_other_adversary, self_X], dim=1)  
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        #print(f'out size:{out.size()}') 
        return out

class RNNMLPNetwork_Policy_Adversary(nn.Module):
    def __init__(self, agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, shuffle=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(RNNMLPNetwork_Policy_Adversary, self).__init__()

        self.agent_id = agent_id
        self.n_agent_ori = n_agent_ori
        self.n_adversary_ori = n_adversary_ori
        self.n_agent = n_agent
        self.n_adversary = n_adversary
        self.n_neurons = input_dim
        self.shuffle = shuffle

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        
        
        self.rnn_agent = nn.RNN(2+2, self.n_agent_ori*(2+2)) #obs
        self.rnn_other_adversary = nn.RNN(2, 2*(self.n_adversary_ori-1))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
        
    
    def init_hidden(self, n_neurons, batch_size, device):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, batch_size, n_neurons, device=device))
    
    def split(self, X):#X is obs:[self_vel, self_pos, landmark_pos, other_pos, other_vel] (16)
        L = X.size(1)
        batch_size = X.size(0)
        # self is adversary
        n_landmark=2
        self_X = X[:, :2+2+2*n_landmark]
        agent_X = X[:, 2+2+2*n_landmark+(self.n_adversary-1)*2:]
       
        agent_pos = agent_X[:,:int(agent_X.size(1)/2)]
        agent_vel = agent_X[:,int(agent_X.size(1)/2):]
        agent_pos = agent_pos.view([batch_size,int(agent_pos.size(1)/2),2])
        agent_vel = agent_vel.view([batch_size,int(agent_vel.size(1)/2),2])
        agent_X = torch.cat([agent_pos, agent_vel], dim=2)
                                
                            
        other_adversary_X = X[:,2+2+2*n_landmark : 2+2+2*n_landmark+2*(self.n_adversary-1)]
        other_adversary_X = other_adversary_X.view([batch_size, int(other_adversary_X.size(1)/2),2])
        
        return agent_X, other_adversary_X, self_X
        
    def forward(self, X):# X is obs:[self_vel, self_pos, landmark_pos, other_pos, other_vel] (16)
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        batch_size = X.size(0)
        #print(f'x size:{X.size()}')
        if(self.shuffle):
            idx = torch.randperm(self.n_neurons)
            X = X[:, idx]
        X = self.in_fn(X)
        agent_X, other_adversary_X, self_X = self.split(X)
        agent_X = agent_X.permute(1, 0, 2)
        other_adversary_X = other_adversary_X.permute(1,0,2)

        #print(f'x size after reshape:{X.size()}')
        self.hidden_agent = self.init_hidden((2+2)*self.n_agent_ori, batch_size, device=X.device)
        self.hidden_other_adversary = self.init_hidden(2*(self.n_adversary_ori-1), batch_size, device=X.device)
        #print(f'hidden size before rnn:{self.hidden.size()}') 
        _, self.hidden_agent = self.rnn_agent(agent_X, self.hidden_agent)
        _, self.hidden_other_adversary = self.rnn_other_adversary(other_adversary_X, self.hidden_other_adversary)
        #print(f'hidden size after rnn:{self.hidden.size()}')  
        self.hidden_agent = self.hidden_agent.view([batch_size, -1])
        self.hidden_other_adversary = self.hidden_other_adversary.view([batch_size,-1])
        X = torch.cat([self.hidden_agent, self.hidden_other_adversary, self_X], dim=1)  
        #print(self.hidden_agent.size(), self.hidden_other_adversary.size(), self_X.size())
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        #print(f'out size:{out.size()}') 
        return out

class RNNMLPNetwork_Critic_Agent(nn.Module):
    def __init__(self, agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, shuffle=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(RNNMLPNetwork_Critic_Agent, self).__init__()

        self.agent_id = agent_id
        self.n_agent_ori = n_agent_ori
        self.n_adversary_ori = n_adversary_ori
        self.n_agent = n_agent
        self.n_adversary = n_adversary
        self.n_neurons = input_dim
        self.shuffle = shuffle

        self.n_landmark = 2
        self.ag_obs_len = 2+2+2*self.n_landmark+(2+2)*(self.n_agent-1)+2*self.n_adversary
        self.ad_obs_len = 2+2+2*self.n_landmark+2*(self.n_adversary-1)+(2+2)*self.n_agent
        self.ag_act_len = 2
        self.ad_act_len=2

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        
        
        #self.rnn_agent = nn.RNN(2, 2*self.n_agent_ori) #o-a pair
        self.rnn_adversary = nn.RNN(self.ad_obs_len+self.ad_act_len, (self.ad_obs_len+self.ad_act_len)*self.n_adversary_ori)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
        
    
    def init_hidden(self, n_neurons, batch_size, device):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, batch_size, n_neurons, device=device))
    
    def split(self, X):
        L = X.size(1)
        batch_size = X.size(0)
        # self is agent
        n_landmark = self.n_landmark
        ag_obs_len = self.ag_obs_len
        ad_obs_len = self.ad_obs_len
        ag_act_len = self.ag_act_len
        ad_act_len = self.ad_act_len
        
        total_obs = ad_obs_len*self.n_adversary+ag_obs_len*self.n_agent
        X_obs, X_act = X[:, :total_obs], X[:,total_obs:]
        agent_obs = X_obs[:, self.n_adversary*ad_obs_len : self.n_adversary*ad_obs_len + self.n_agent*ag_obs_len] 
        agent_act = X_act[:, self.ag_act_len*self.n_adversary:self.ag_act_len*self.n_adversary+self.ag_act_len*self.n_agent]
        agent_X = torch.cat([agent_obs, agent_act], dim=1)
       
        adversary_obs = X_obs[:, : self.n_adversary*ad_obs_len]   
        adversary_obs = adversary_obs.view([batch_size, self.n_adversary, -1])
        adversary_act = X_act[:, : self.n_adversary*ad_act_len]
        adversary_act = adversary_act.view([batch_size, self.n_adversary, -1])
        adversary_X = torch.cat([adversary_obs, adversary_act], dim=2)
    
        return agent_X, adversary_X
        
    def forward(self, X):# X[o0,o1,...,on, a0, a1,...,an]
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        batch_size = X.size(0)
        #print(f'x size:{X.size()}')
        if(self.shuffle):
            idx = torch.randperm(self.n_neurons)
            X = X[:, idx]
        X = self.in_fn(X)
        agent_X, adversary_X = self.split(X)
        #agent_X = agent_X.permute(1, 0, 2)
        adversary_X = adversary_X.permute(1,0,2)

        #print(f'x size after reshape:{X.size()}')
        #self.hidden_agent = self.init_hidden(2*self.n_agent_ori, batch_size, device=X.device)
        self.hidden_adversary = self.init_hidden((self.ad_obs_len+self.ad_act_len)*self.n_adversary_ori, batch_size, device=X.device)
        #print(f'hidden size before rnn:{self.hidden.size()}') 
        #_, self.hidden_agent = self.rnn_agent(agent_X, self.hidden_agent)
        _, self.hidden_adversary = self.rnn_adversary(adversary_X, self.hidden_adversary)
        #print(f'hidden size after rnn:{self.hidden.size()}')  
        #self.hidden_agent = self.hidden_agent.view([batch_size, -1])
        self.hidden_adversary = self.hidden_adversary.view([batch_size,-1])
        X = torch.cat([agent_X, self.hidden_adversary], dim=1)  
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        #print(f'out size:{out.size()}') 
        return out

class RNNMLPNetwork_Policy_Agent(nn.Module):
    def __init__(self, agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, shuffle=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(RNNMLPNetwork_Policy_Agent, self).__init__()

        self.agent_id = agent_id
        self.n_agent_ori = n_agent_ori
        self.n_adversary_ori = n_adversary_ori
        self.n_agent = n_agent
        self.n_adversary = n_adversary
        self.n_neurons = input_dim
        self.shuffle = shuffle

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        
        
        self.rnn_agent = nn.RNN(1, self.n_agent_ori) #obs
        self.rnn_adversary = nn.RNN(2, 2*self.n_adversary_ori)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
        
    
    def init_hidden(self, n_neurons, batch_size, device):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, batch_size, n_neurons, device=device))
    
    def split(self, X):
        L = X.size(1)
        batch_size = X.size(0)
        # self is agent
        n_landmark=2
        self_X = X[:, :2+2+2*n_landmark]
        other_agent_X = X[:, 2+2+2*n_landmark+self.n_adversary*2:]
        agent_X = torch.cat([self_X, other_agent_X], dim=1)
                                
                            
        adversary_X = X[:,2+2+2*n_landmark : 2+2+2*n_landmark+2*self.n_adversary]
        adversary_X = adversary_X.view([batch_size, int(adversary_X.size(1)/2),2])
        return agent_X, adversary_X
        
    def forward(self, X):# X[o0,o1,...,on]
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        batch_size = X.size(0)
        #print(f'x size:{X.size()}')
        if(self.shuffle):
            idx = torch.randperm(self.n_neurons)
            X = X[:, idx]
        X = self.in_fn(X)
        agent_X, adversary_X = self.split(X)
        adversary_X = adversary_X.permute(1,0,2)

        #print(f'x size after reshape:{X.size()}')
        # self.hidden_agent = self.init_hidden(self.n_agent_ori, batch_size, device=X.device)
        self.hidden_adversary = self.init_hidden(2*self.n_adversary_ori, batch_size, device=X.device)
        #print(f'hidden size before rnn:{self.hidden.size()}') 
        # _, self.hidden_agent = self.rnn_agent(agent_X, self.hidden_agent)
        _, self.hidden_adversary = self.rnn_adversary(adversary_X, self.hidden_adversary)
        #print(f'hidden size after rnn:{self.hidden.size()}')  
        # self.hidden_agent = self.hidden_agent.view([batch_size, -1])
        self.hidden_adversary = self.hidden_adversary.view([batch_size,-1])
        X = torch.cat([agent_X, self.hidden_adversary], dim=1)  
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        #print(f'out size:{out.size()}') 
        return out
     