import numpy as np


class ReplayBuffer:

    def __init__(self, capacity, state_shape=(4, 84, 84)):
        self._capacity = capacity
        self._ptr = 0
        self._size = 0

        # Pre-allocate fixed arrays — avoids repeated memory allocation
        self._states      = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self._actions     = np.zeros((capacity,),              dtype=np.int64)
        self._rewards     = np.zeros((capacity,),              dtype=np.float32)
        self._next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self._dones       = np.zeros((capacity,),              dtype=bool)

    def store(self, state, action, reward, next_state, done):
        # TODO: write each field at index self._ptr
        self._states[self._ptr] = state
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr] = done 
        # TODO: advance self._ptr with wrap-around (% self._capacity)
        self._ptr = (self._ptr + 1)% self._capacity
        # TODO: update self._size (capped at self._capacity)
        self._size = min(self._size + 1 ,self._capacity)
        

    def sample(self, batch_size):
        # TODO: sample batch_size random indices from valid range [0, self._size)
        indices = np.random.randint(0, self._size, size=batch_size)
        # TODO: return states and next_states as float32 / 255
        return (
            self._states[indices].astype(np.float32) / 255.0,
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices].astype(np.float32) / 255.0,
            self._dones[indices],
        )
        

    def __len__(self):
        return self._size
