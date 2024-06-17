import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    '''
    Input: State
    Output: State-Action value for each action'''
    def __init__(self, state_size: int, action_size: int) -> None:
        """
        Initialize the QNet model.

        Args:
            state_size (int): The size of the state.
            action_size (int): The size of the action.
        """
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, state_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size).

        This method takes an input tensor `x` of shape (batch_size, state_size) and performs a forward pass through the neural network. It applies the ReLU activation function to each layer and returns the output tensor of shape (batch_size, action_size).
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class QAgent:
    '''
    practical11_DQN.ipynb
    '''
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 epsilon_decay: float,
                 epsilon_min: float,
                 memory_size: int,
                 ):
        """
        :param state_size: environment state size.
        :param action_size: number of action in which we want to discretize the action space.
        :param epsilon_decay: epsilon decay for epsilon-greedy policy.
        :param epsilon_min: minimum value of epsilon.
        :param memory_size: max length of double ended queue replay buffer memory.
        """
        self.state_size = state_size
        self.action_size = action_size

        # replay buffer memory
        # TRICK 1 -> add and remove from both ends is O(1), fixed size with automatic overflows, random access
        self.memory = deque(maxlen=memory_size)     

        # epsilon
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # hyperparameters
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.0005  # Learning rate used to update network
        self.checkpoint_episode = 50  # Used to save network
        self.tau = 0.005  # TRICK 2 -> Tau=Beta used to update target network -> Beta of W_ = W_ + beta* (W - W_)

        # for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.returns = []
        self.steps = []

        # Initialize neural networks
        self.model = QNet(state_size, action_size).to(self.device)
        self.target_model = QNet(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model weights to match model weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    def update_target_model(self):
        """
        Soft update of the target network's weights: θ′ ← τ θ + (1 − τ )θ′
        """
        # self.target_model.load_state_dict(self.model.state_dict())
        target_state_dict = self.target_model.state_dict()
        state_dict = self.model.state_dict()

        for key in state_dict:
            target_state_dict[key] = state_dict[key]*self.tau + target_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_state_dict)
    

    def remember(self, state, action, reward, next_state, done):
        """
        remember method for replay buffer memory
        """
        self.memory.append((state, action, reward, next_state, done))
    

    def act(self, state, training=True):
        """
        act method for epsilon greedy policy
        """
        state = torch.from_numpy(state).float().to(device)
        
        if np.random.rand() <= self.epsilon and training:
            return random.randrange(self.action_size)
        
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()
    

    def replay(self, batch_size):
        """
        A method to replay experiences stored in memory and update the Q-values of the model.
        
        :param batch_size: The number of experiences to replay.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().to(device)
            next_state = torch.from_numpy(next_state).float().to(device)
            reward = torch.tensor(reward).float().to(device)

            # update Q value
            if done:
                y = reward
            else:
                y = reward + self.gamma * torch.max(self.target_model(next_state).detach()) # detach a tensor from the computation graph
            
            y_hat = self.model(state)[0][action]     # Q(s,a)
            loss = self.criterion(y, y_hat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def save_model(self, directory: str):
        torch.save(self.model.state_dict(), directory)


    def training(self)