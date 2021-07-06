import torch
import torch.nn.functional as F

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def whiten(values):
    """Whiten values."""
    mean, std = torch.mean(values), torch.std(values)
    whitened = (values - mean) / (std + 1e-8)
    return whitened
    #mu = torch.mean(values,dim=-1,keepdim=True)
    #sd = torch.std(values,dim=-1,keepdim=True)
    #return (values - mu)/sd

def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd*logits, axis=-1)
    return entropy
