import numpy as np

# create replay buffer of tuples of (state, action, reward, next_state, done)
class MemoryBuffer():
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """
        Adds data to the memory buffer.

        Parameters:
            data (Any): The data to be added to the memory buffer.

        Returns:
            None

        This function adds the given data to the memory buffer. If the buffer is already full, it replaces the oldest
        data with the new data. Otherwise, it appends the data to the buffer.

        Note:
            - The memory buffer has a maximum size specified by the `max_size` attribute.
            - The `ptr` attribute keeps track of the position of the next data to be added or replaced.
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the memory buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing numpy arrays of the following:
                - state (ndarray): An array of states.
                - action (ndarray): An array of actions.
                - reward (ndarray): An array of rewards.
                - next_state (ndarray): An array of next states.
                - done (ndarray): An array of booleans indicating if the episode is done.
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in ind: 
            STATE, ACTION, REWARD, NEXT_STATE, DONE = self.storage[i]
            state.append(np.array(STATE, copy=False))
            action.append(np.array(ACTION, copy=False))
            reward.append(np.array(REWARD, copy=False))
            next_state.append(np.array(NEXT_STATE, copy=False))
            done.append(np.array(DONE, copy=False))

        return np.array(state), np.array(action), np.array(reward).reshape(-1,1), np.array(next_state), np.array(done).reshape(-1,1)
    