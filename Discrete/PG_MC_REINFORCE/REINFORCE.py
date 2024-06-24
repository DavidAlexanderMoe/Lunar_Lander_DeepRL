import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size: int=256):
        super(ActorNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.clamp(x,-1.1,1.1)
        x = F.relu(self.dense_layer_1(x))
        x = F.relu(self.dense_layer_2(x))
        return F.softmax(self.output(x),dim=-1) #-1 to take softmax of last dimension
    

class ValueFunctionNet(nn.Module):
    def __init__(self, state_size, hidden_size: int=256):
        super(ValueFunctionNet, self).__init__()
        self.dense_layer_1 = nn.Linear(state_size, hidden_size)
        self.dense_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.clamp(x,-1.1,1.1)
        x = F.relu(self.dense_layer_1(x))
        x = F.relu(self.dense_layer_2(x))
        return self.output(x)
    

class MCPGAgent():
    '''
    REINFORCE Monte Carlo Policy Gradient Agent with Value Function Approximation as Baseline
    '''
    def __init__(self, state_size, action_size, actor_lr, vf_lr, discount, hidden_size: int=256):
        self.action_size = action_size
        self.actor_net = ActorNet(state_size, action_size, hidden_size).to(device)
        self.vf_net = ValueFunctionNet(state_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.vf_optimizer = optim.Adam(self.vf_net.parameters(), lr=vf_lr)
        self.discount = discount
        
    def act(self, state):
        """
        Given a state, this function takes in a state and returns an action. 
        It first computes the action probabilities using the actor network. 
        Then it randomly samples an action from the probabilities using numpy's random choice function. 
        The action is then returned.
        
        :param state: A numpy array representing the current state of the environment.
        :type state: numpy.ndarray
        :return: An integer representing the chosen action.
        :rtype: int
        """
        # get action probs then randomly sample from the probabilities
        with torch.no_grad():
            input_state = torch.FloatTensor(state).to(device)
            # input_state = torch.from_numpy(state).float().to(device)
            action_probs = self.actor_net(input_state).detach().cpu().numpy()
            # detach and turn to numpy to use with np.random.choice() with the probabilities
            action = np.random.choice(np.arange(self.action_size), p=action_probs)
        return action

    def train(self, states, actions, rewards):
        """
        Trains the agent using the given state, action, and reward lists.

        Args:
            states (List[List[float]]): A list of state vectors, where each state vector represents the state of the environment at a given time step.
            actions (List[int]): A list of integers representing the actions taken at each time step.
            rewards (List[float]): A list of floats representing the rewards received at each time step.

        Returns:
            Tuple[float, float]: A tuple containing the actor loss and the value function loss.

        """

        # turn rewards into return
        G_array = np.zeros((len(rewards),))
        G = 0.

        # calculate the return in reverse order to take into account the delayed rewards and assign credit to the actions that led to those rewards.
        # + improve the efficiency and stability of learning due to variance reduction of the estimated returns -> learning could be more stable and efficient
        # This is because the earlier time steps have more data points to estimate the return, while the later time steps have fewer data points due to the delayed rewards.
        for i in range(len(rewards) - 1, -1, -1):
            # G = r + gamma*G = sum_k=t+1 (gamma^(k-t-1) * R_k)
            G = rewards[i] + self.discount*G
            G_array[i] = G
            
        # create tensors
        state = torch.FloatTensor(states).to(device)
        # convert action list to Long type (more efficient than Float for discrete values) + view to reshape the tensor to have one column and as many rows as necessary
        action = torch.LongTensor(actions).to(device).view(-1,1)
        returns = torch.FloatTensor(G_array).to(device).view(-1,1)
        
        # get value function estimates
        vf = self.vf_net(state).to(device)
        with torch.no_grad():
            # compute delta = G - V(s) (REINFORCE with Baseline)
            # delta = TD error
            delta = returns - vf
        
        # calculate actor loss
        selected_action_prob = self.actor_net(state).gather(1, action)      # self.model(state)[0][action]
        
        # REINFORCE loss:
        #actor_loss = torch.mean(-torch.log(selected_action_prob) * return)
        
        # https://pytorch.org/docs/stable/distributions.html

        # REINFORCE with Baseline loss:
        # Update: θ <- θ + alpha * gamma^t * delta *gradient(log(pi(A|S, θ)))
        actor_loss = torch.mean(-torch.log(selected_action_prob) * delta)
        
        self.actor_optimizer.zero_grad()   # clear gradient for next iteration
        actor_loss.backward()              # backprop
        self.actor_optimizer.step()        # update θ

        # calculate vf loss
        # update: w <- w + alpha * delta * gradient(V(S, w))
        loss_fn = nn.MSELoss()
        vf_loss = loss_fn(vf, returns)
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()            # update w
        
        return actor_loss.detach().cpu().numpy(), vf_loss.detach().cpu().numpy()        # return on CPU and transform in numpy array