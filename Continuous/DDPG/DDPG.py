import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from memory import MemoryBuffer

# check and use GPU if available if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The deterministic policy gradients are used to train the actor network. 
# DDPG can be (it is here) off-policy and combines some methods used for DQN like target network updating and an experience replay buffer. 
# Unlike DQN, DDPG can be used in the continuous action space.
    
class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_max):
        """
        Initializes the ActorNet object.
        The Actor network is trained to learn a policy that maps states to actions, with the goal of maximizing the expected return.

        Args:
            state_size (int): The size of the input state.
            action_size (int): The size of the output action.
            hidden_size (int): The size of the hidden layers.
            action_max (float): The maximum value of the action.
       
        Inputs:
            state (torch.Tensor): The current state of the environment.

        Returns:
            mu(s,θ) -> action (torch.Tensor): The action to be taken by the agent in that state.
        """
        super(ActorNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)
        self.action_max = action_max
        # not necessary since the action space is bounded in [-1, 1]
        # therefore action_max = 1.0
    
    def forward(self, x):
        '''
        torch.clamp clamps the input state to the range [-1.1, 1.1]. 
        This is done to ensure that the input to the network is within a reasonable range.
        The tanh function scales the output to the range [-1, 1]. 
        Finally, the output is multiplied by action_max to scale it to the appropriate range for the action space 
        (otherwise the actions that the agent takes may be limited to a small fraction of the full range of possible actions)
        The resulting vector represents the action that the agent should take in the current state.        
        '''
        x = torch.clamp(x,-1.1,1.1)
        x = F.relu(self.dense_layer_1(x))
        x = F.relu(self.dense_layer_2(x))
        return torch.tanh(self.output(x)) * self.action_max
        # return torch.tanh(self.output(x))
    
class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        """
        Initializes the CriticNet object.
        The Critic network is trained to learn a mapping from state-action pairs to Q-values, which can then be used to update the Actor network's policy.

        Parameters:
            state_size (int): The size of the state.
            action_size (int): The size of the action.
            hidden_size (int): The size of the hidden layers.

        Inputs:
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken in that state.
            Both needed to compute Q(s,a)

        Returns:
            Q-value -> (torch.Tensor): The Q-value of the state-action pair (1 single value).
            Q = expected return that the agent will receive if it takes that action in that state.
        """
        super(CriticNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size+action_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x, a):
        '''
        torch.cat((x,a),dim=1) := if x is a tensor of shape (batch_size, state_size) and a is a tensor of shape (batch_size, action_size), 
        then the resulting tensor will have a shape of (batch_size, state_size + action_size).
        This is done in the Critic network to combine the state and action information into a single input vector, 
        which can then be processed by the network to estimate the Q-value of the state-action pair.
        '''
        x = torch.clamp(x,-1.1,1.1)
        x = F.relu(self.dense_layer_1(torch.cat((x,a),dim=1)))
        x = F.relu(self.dense_layer_2(x))
        return self.output(x)


class DDPGAgent():
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 actor_lr, 
                 critic_lr, 
                 discount,
                 min_action, 
                 max_action, 
                 exploration_noise):
        
        self.state_size = state_size
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.exploration_noise = exploration_noise
        self.discount = discount
        self.criterion = nn.MSELoss()

        self.actor = ActorNet(state_size, action_size, hidden_size, max_action).to(device)
        self.actor_target = ActorNet(state_size, action_size, hidden_size, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticNet(state_size, action_size, hidden_size).to(device)
        self.critic_target = CriticNet(state_size, action_size, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        
    def act(self, state, add_noise=True):
        #get action probs then randomly sample from the probabilities
        with torch.no_grad():
            input_state = torch.FloatTensor(state).to(device)
            action = self.actor(input_state)              # get action from actor and preprocessed input state
            action = action.detach().cpu().numpy()        # detach and turn to numpy to use with np.random.choice()

            if add_noise:
                # add exploration noise to action 
                noise = np.random.normal(0., self.exploration_noise, size=self.action_size)
                action = (action + noise).clip(self.min_action, self.max_action)
                # in DDPG add noise for exploration as sigma = degree of exploration, then clip the resulting distribution to the min and max action
                # to ensure that the action space is between the range of possible actions in the environment 
        return action

    
    def train(self, memory_buffer, batch_size):
        '''
        Algorithm: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        '''
        # sample a batch from the replay buffer
        state, action, reward, next_state, done = memory_buffer.sample(batch_size)
        
        # preprocess to turn batches into tensors and use GPU if available
        # these are all batches of data
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)       # should be 1-done but will do it in the training loop

        # get target network target values
        with torch.no_grad():

            # obtain the target action from the actor target network 
            # a = mu(s,θ) and resize it to (batch_size, action_size)
            target_action = self.actor_target(next_state).view(batch_size,-1)

            # compute the targets: y(r, s', d) = r + γ * (1 - d) * Q_targ(s', mu_targ(s'))
            # y = reward + (1 - done) * self.discount * self.critic_target(next_state, mu_targ).view(batch_size, -1)
            target_value = reward + done * self.discount * self.critic_target(next_state, target_action).view(batch_size, -1)
        
        # get training network values for updating the critic network
        # Q = Q_train(s, a)
        critic_value = self.critic(state, action).view(batch_size, -1)
        
        # train critic with backprop
        critic_loss = F.smooth_l1_loss(critic_value, target_value)   # more robust to outliers
        # critic_loss = self.criterion(critic_value, target_value)   # MSE Loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()         # update w
        
        # train actor with backprop
        train_action = self.actor(state)
        # loss taken from https://proceedings.mlr.press/v32/silver14.pdf
        actor_loss = -torch.mean(self.critic(state, train_action))      # -1 * Q_train(s, mu(s,θ))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()          # update θ
        
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
        
    
    def update_target_network_soft(self, num_iter, update_tau=0.001):
        # soft target network update: update target networks with mixture of train and target
        if num_iter % 1 == 0:       # update at each iteration (1)
            for target_var, var in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_var.data.copy_((1.-update_tau) * target_var.data + (update_tau) * var.data)
            for target_var, var in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_var.data.copy_((1.-update_tau) * target_var.data + (update_tau) * var.data)

