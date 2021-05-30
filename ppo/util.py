import torch

from constants import *

# Expect values to be one longer than rewards
def get_targets_and_advs(rewards, values):
    td_target = rewards + GAMMA * values[1:] # reward + value of next state
    td_target[-1] = rewards[-1] # Remove discounted future reward from terminal
    
    delta = td_target - values[:-1] # Difference between target and predicted value

    advantages = torch.zeros_like(delta)
    adv = 0.0
    N = len(delta)
    for t in reversed(range(0, N)):
        adv = GAMMA * TAU * adv + delta[t]
        advantages[t] = adv

    return td_target, normalize(advantages)

# Normalize tensor by its mean and std
def normalize(t):
    return (t - t.mean()) / (t.std() + 1e-8)

def makeMask(done):
    mask = torch.tensor([1.0 - done], device = 'cuda')
    return mask

def generate_indices(total_size, batch_size):
    inds = torch.randperm(total_size)
    return [inds[i * batch_size:(i+1) * batch_size] for i
            in range(0, total_size // batch_size)]

def reduce_tensor(t):
    if type(t) is torch.Tensor:
        return t.squeeze()
    else:
        return t

class RolloutStorage:
    def  __init__(self, MAX_SIZE = 4096):
        # Using empty tensors makes it easier to assert
        # shape on insertions
        self.store = None
        self.size = None
        self.MAX_SIZE = MAX_SIZE
        self.reset()

    def remember(self, log_prob, value, state, action, reward):
        if self.size == self.MAX_SIZE:
            return
        size = self.size

        self.store['log_prob'][size] = reduce_tensor(log_prob)
        self.store['value'][size] = reduce_tensor(value)
        self.store['state'][size] = reduce_tensor(state)
        self.store['action'][size] = reduce_tensor(action)
        self.store['reward'][size] = reduce_tensor(reward)
        self.size += 1

    def detach(self):
        for key in self.store.keys():
            self.store[key] = self.store[key].detach()

    def cuda(self):
        for key in self.store.keys():
            self.store[key] = self.store[key].to('cuda')

    def reset(self):
        MAX_SIZE = self.MAX_SIZE
        self.store = {'log_prob' : torch.zeros(MAX_SIZE),
                'value' : torch.zeros(MAX_SIZE),
                'state' : torch.zeros(MAX_SIZE, STATE_DIM), 
                'action' : torch.zeros(MAX_SIZE),
                'reward' : torch.zeros(MAX_SIZE)}
        self.size = 0

    def set_terminal(self, last_v):
        self.store['value'][self.size] = last_v

    def get(self, key):
        if key == 'value':
            return self.store['value'][0:self.size+1]
        else:
            return self.store[key][0:self.size]
