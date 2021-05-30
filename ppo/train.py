import gym
from models import BaseLinePPO
import torch
from tqdm import tqdm
import util
import ppo
from constants import LEARNING_RATE

env = gym.make('CartPole-v0')
agent = BaseLinePPO(4, 2)
agent.cuda()

storage = util.RolloutStorage()

opt = torch.optim.Adam(agent.parameters(), lr = LEARNING_RATE)

TIME_LIMIT = 9999
EPISODES = 1000

def prep_state(s):
    s = torch.from_numpy(s).float()
    s = s.to('cuda').unsqueeze(0)
    return s

for episode in tqdm(range(EPISODES)):
    agent.eval()
    s = env.reset()
    s = prep_state(s)
    total_r = 0

    for t in range(TIME_LIMIT):
        pi, v, a = agent(s)
        v = v[0].item()
        log_prob = pi.log_prob(a)

        env.render()
        s_next, r, done, info = env.step(a.item())
        s_next = prep_state(s_next)
        total_r += r
        
        storage.remember(log_prob, v, s, a, r)
        
        if done: break

        s = s_next

    _, next_v, _ = agent(s_next)
    storage.set_terminal(next_v)

    loss = ppo.train_PPO(agent, opt, storage)
    storage.reset()
    print("EPISODE " + str(episode) + "|Loss " + str(round(loss, 2)) + "|REWARD " + str(total_r))
env.close()