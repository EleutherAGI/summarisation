import torch
from utils import logprobs_from_logits, normalize, clip_by_value, entropy_from_logits

class PPO:
    def __init__(self, tokenizer, model, optimizer, wandb=None):

        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer

        self.value_clip = .2
        self.advantage_clip = .2
        self.gamma = 1
        self.lam = 0.95
        self.critic_coef = 1
        self.ppo_epochs = 3

        self.wandb = wandb
        if self.wandb:
            wandb.watch(self.model)


    def step(self, logprobs, values, rewards, model_input, per_device_batch_size=2, gradient_acc_steps=4):
        large_batch_size = logprobs.shape[0]
        for _ in range(self.ppo_epochs):
            idxs = torch.randperm(large_batch_size)

            self.optimizer.zero_grad()
            for i in range(int(large_batch_size/per_device_batch_size)):
                idx = idxs[i*per_device_batch_size:(i+1)*per_device_batch_size]

                loss_p, loss_v = self.calculate_loss(logprobs[idx], 
                                                     values[idx], 
                                                     rewards[idx], 
                                                     model_input[idx])
                loss = loss_p + loss_v
                loss.backward()
                
                if (i+1)%gradient_acc_steps == 0: 

                    self.optimizer.step()
                    self.optimizer.zero_grad()




    def calculate_loss(self, old_logprobs, values, rewards, model_input, gen_len = 32):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = normalize(advantages)
        advantages = advantages.detach()

        model_outputs = self.model(model_input)
        logits = model_outputs['logits']
        vpred = model_outputs['values']


        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1, 0]

        vpredclipped = clip_by_value(vpred,
                                     values - self.value_clip,
                                     values + self.value_clip)

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.advantage_clip,
                                               1.0 + self.advantage_clip)

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.critic_coef * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))

        if self.wandb:
            self.wandb.log({
                "value_loss": vf_loss,
                "policy_loss": pg_loss,
                "loss": loss,
                "entropy": entropy,
                "returns": returns,
                "value_clipfrac": vf_clipfrac,
                "clipfrac": pg_clipfrac,
                "kl-update": torch.mean(logprob - old_logprobs),
            })

        return pg_loss, self.critic_coef * vf_loss