import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time
import gymnasium as gym
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    '''
    Input: State
    Output: State-Action value for each action'''
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize the QNet model.

        Args:
            state_size (int): The size of the state.
            action_size (int): The size of the action.
        """
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

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
        return self.fc3(x)



class QAgent:
    def __init__(self,
                 environment: gym.Env):
        """
        :param environment: The environment to train the agent in.
        """
        self.state_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n

        # replay buffer memory
        # TRICK 1 -> add and remove from both ends is O(1), fixed size with automatic overflows, random access
        self.memory = deque(maxlen=20_000)

        # epsilon
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998

        # hyperparameters
        self.gamma = 0.99               # Discount factor
        self.learning_rate = 0.0005     # Learning rate used to update network
        self.tau = 0.005                # TRICK 2 -> Tau=Beta used to update target network -> Beta of W_ = W_ + beta* (W - W_)
        self.checkpoint_episode = 20

        # Initialize neural networks
        self.model = QNet(self.state_size, self.action_size).to(device)
        self.target_model = QNet(self.state_size, self.action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())       # Initialize target model weights to match model weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.returns = []
        self.avg_scores_array = []
        self.tot_steps = []

    def update_target_model(self):
        """
        Soft update of the target network's weights: θ′ ← τ θ + (1 − τ )θ′
        """
        # self.target_model.load_state_dict(self.model.state_dict())
        target_state_dict = self.target_model.state_dict()
        state_dict = self.model.state_dict()

        for key in state_dict:
            target_state_dict[key] = state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
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
        # torch.no_grad() for 3 reasons:
            # Efficiency: Saves memory and computational resources by not calculating gradients.
            # Inference Mode: Indicates that the operations within the block are for inference, not training.
            # Gradient Isolation: Ensures that action selection doesn't affect the training process.
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)

            # With probability epsilon, select a random action (exploration)
            if np.random.rand() <= self.epsilon and training:
                return random.randrange(self.action_size)
            
            # With probability 1-epsilon, select the action with the highest value (exploitation)
            act_values = self.model(state)                  # Perform a forward pass to get (state, action) values for every action
            return torch.argmax(act_values, dim=1).item()   # Select the action with the highest value and return it
    

    def replay(self, batch_size):
        """
        method to replay experiences stored in memory and update the Q-values of the model.
        
        :param batch_size: The number of experiences to replay.
        """
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().to(device)
            next_state = torch.from_numpy(next_state).float().to(device)
            reward = torch.tensor(reward).float().to(device)

            # update target U for backprop
            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(self.target_model(next_state).detach()) # detach a tensor from the computation graph
            
            Q_sa = self.model(state)[0][action]
            # self.model(state) outputs a (1 x action size) matrix (since it is a for loop)
            # otherwise it would output a batch_size x action size matrix
            # you select the first and only sublist 
            # and then [action] is used to select the estimated state-action value for the current action -> Q value for that (s,a) pair
            
            # loss backprop
            loss = self.criterion(target, Q_sa)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save_model(self, 
                   directory: str):
        torch.save(self.model.state_dict(), directory)



    def training(self,
                 env: gym.Env,
                 batch_size: int,
                 episodes: int,
                 steps: int = 1000,
                 directory: str = '..\Trained_Agents') -> list[float]:
        """
        method to train the agent.
        this method outputs a list of float numbers <- returns
        
        :param env: The environment to train the agent in.
        :param batch_size: The number of experiences to replay.
        :param episodes: The number of episodes to train the agent for.
        :param directory: The directory to save the trained agent in.
        """
        # Create directories
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        if not os.path.exists(os.path.join(directory, f'returns')):
            os.makedirs(os.path.join(directory, f'returns'))
        
        # episode loop
        for episode in range(1, episodes + 1):
            done = False
            tot_reward = 0
            steps =  0
            
            state, _ = env.reset()
            state = np.reshape(state, [1, self.state_size])

            # steps loop
            while not done:
                # env.render()

                # Choose action
                action = self.act(state)

                # Take action
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                # cumulative reward and steps
                tot_reward += reward
                steps += 1

                # Store transition (s,a,r,s') into memory
                self.remember(state, action, reward, next_state, done)
                
                # Update current state to new one
                state = next_state

                if done:
                    reward = -1
                    print(f"episode: {episode}/{episodes}, return: {tot_reward}, epsilon: {self.epsilon:.2f}, steps: {steps}")
                    break

                # Experience replay trick for convergence issues
                if len(self.memory) > batch_size:
                    self.replay(batch_size=batch_size)
                    self.update_target_model()

            # Store episode returns
            self.returns.append(tot_reward)
            avg_score = np.mean(self.returns)
            self.avg_scores_array.append(avg_score)
            self.tot_steps(steps)
            
            # Save checkpoint model
            if episode % 500 == 0:
                self.save_model(os.path.join(directory, f'QAgent_ep{episode}.pth'))

        # save the full model and returns
        np.save(os.path.join(directory, f'returns\QAgent_returns.npy'), np.array(self.returns))
        np.save(os.path.join(directory, f'returns\QAgent_avg_scores_array.npy'), np.array(self.avg_scores_array))
        self.save_model(os.path.join(directory, f'QAgent_final.pth'))
        return self.returns
    

    def load(self, filename):
        self.model.load_state_dict(torch.load(f'..\\Trained_Agents\\{filename}_final.pth'))

    
    def testing(self, 
                env: gym.Env, 
                episodes: int):
        state, _ = env.reset()
        scores_deque = deque(maxlen=100)

        for i_episode in range(1, episodes + 1):
            state, _ = env.reset()
            state = np.reshape(state, [1, self.state_size])

            total_reward = 0
            time_start = time.time()
            timesteps =  0

            while True:
                env.render()

                action = self.act(state, training=False)

                next_state, reward, done, _, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                state = next_state
                total_reward += reward
                timesteps += 1

                if done:
                    break 

            delta = (int)(time.time() - time_start)

            scores_deque.append(total_reward)

            print('Episode {}\tAverage Score: {:.2f}, \t Timesteps: {} \tTime: {:02}:{:02}:{:02}'\
                      .format(i_episode, np.mean(scores_deque), timesteps,\
                              delta//3600, delta%3600//60, delta%60))