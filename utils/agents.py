#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-03-19 14:40:19
@LastEditTime: 2019-03-20 22:23:14
'''
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork, RNNMLPNetwork_Critic_Adversary, RNNMLPNetwork_Critic_Agent, RNNMLPNetwork_Policy_Adversary, RNNMLPNetwork_Policy_Agent
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, use_rnn=1, shuffle=False,
                 agent_id=0, is_adversary=1, n_agent_ori=1, n_adversary_ori=3, n_agent=1, n_adversary=3):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        #print('agent_id:', agent_id, is_adversary)
        if(use_rnn):
            if(is_adversary):
                print('num_in_pol',num_in_pol)
                self.policy = RNNMLPNetwork_Policy_Adversary(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            constrain_out=True,
                                            discrete_action=discrete_action, shuffle=shuffle)
                self.critic = RNNMLPNetwork_Critic_Adversary(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_critic, 1,
                                            hidden_dim=hidden_dim,
                                            constrain_out=False, shuffle=shuffle)
                self.target_policy = RNNMLPNetwork_Policy_Adversary(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            constrain_out=True,
                                            discrete_action=discrete_action, shuffle=shuffle)
                self.target_critic = RNNMLPNetwork_Critic_Adversary(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_critic, 1,
                                            hidden_dim=hidden_dim,
                                            constrain_out=False, shuffle=shuffle)
            else:
                self.policy = RNNMLPNetwork_Policy_Agent(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            constrain_out=True,
                                            discrete_action=discrete_action, shuffle=shuffle)
                self.critic = RNNMLPNetwork_Critic_Agent(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_critic, 1,
                                            hidden_dim=hidden_dim,
                                            constrain_out=False, shuffle=shuffle)
                self.target_policy = RNNMLPNetwork_Policy_Agent(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            constrain_out=True,
                                            discrete_action=discrete_action, shuffle=shuffle)
                self.target_critic = RNNMLPNetwork_Critic_Agent(agent_id, n_agent_ori, n_adversary_ori, n_agent, n_adversary,
                                            num_in_critic, 1,
                                            hidden_dim=hidden_dim,
                                            constrain_out=False, shuffle=shuffle)
                
        else:
            self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
            self.critic = MLPNetwork(num_in_critic, 1,
                                            hidden_dim=hidden_dim,
                                            constrain_out=False)
            self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            constrain_out=True,
                                            discrete_action=discrete_action)
            self.target_critic = MLPNetwork(num_in_critic, 1,
                                            hidden_dim=hidden_dim,
                                            constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
