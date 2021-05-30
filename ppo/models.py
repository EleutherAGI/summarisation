import torch
from torch import nn
from torch.distributions.categorical import Categorical

# Implements a PPO model based on some starting model
# Follows same process as Summarizing From Human Feedback for initialization
# Given a language model as input,
# creates separate policy/actor and critic/value networks from langauge model
class TextGenPPO(nn.Module):
    def __init__(self, start_model, ctx_len):
        super().__init__()

        self.actor = start_model.copy()
        self.critic = start_model.copy()

        self.critic_fc = nn.Linear(start_model.hidden_size, 1)

    # Get policy distribution and value of state x
    # expects state as token sequence (long tensor)
    def forward(self, x):
        act_probs = self.actor(x, return_dict = False, return_pt = True)[0]
        value = self.critic(x, return_dict = False, return_pt = True)[0]
        value = self.critic_fc(value)
        
        pi = Categorical(act_probs)
        token = pi.sample()

        return pi, value, token

class BaseLinePPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
            nn.Softmax(dim = -1))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1))

    def forward(self, x):
        act_probs = self.actor(x)
        value = self.critic(x)

        pi = Categorical(act_probs)
        action = pi.sample()

        return pi, value, action