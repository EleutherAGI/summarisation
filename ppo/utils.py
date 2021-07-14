import torch
import torch.nn.functional as F

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def whiten(values, mask):
    """Whiten values."""
    mean, std = torch.mean(values[mask]), torch.std(values[mask])
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


def query_model(inputs, model, batch_size, response_length = 32):    
    
    tensor_shape, prompt_length = inputs.shape[:2]
    
    response_tensor = torch.full((tensor_shape,prompt_length+response_length), model.config.pad_token_id)
    
    for i in range(int(tensor_shape/batch_size)):
        with torch.no_grad():
            ids = inputs[i*batch_size:(i+1)*batch_size].to(model.device)
            mask = torch.ne(ids, model.config.pad_token_id).long().to(model.device)
            
            generation_output = model.generate(input_ids=ids,
                                               attention_mask=mask,
                                               max_length=prompt_length+response_length, 
                                               do_sample=True, min_length=-1)
            
            response_tensor[i*batch_size:(i+1)*batch_size, :generation_output.shape[1]] = generation_output

            
    generation_mask = torch.zeros_like(response_tensor)
    generation_mask[:, prompt_length:] = torch.ne(response_tensor[:, prompt_length:], model.config.pad_token_id)
    
    return response_tensor, generation_mask

def get_scores(inputs, model, batch_size):
    tensor_shape = inputs.shape[0]
    score_tensor = torch.zeros((tensor_shape,))
    for i in range(int(tensor_shape/batch_size)):
        tokens = inputs[i*batch_size:(i+1)*batch_size].to(model.device)
        tokens = shift_left(tokens)
        mask = torch.ne(tokens, model.config.pad_token_id).long().to(model.device)

        logits = torch.squeeze(model(
            input_ids=tokens,
            attention_mask = mask
        )['logits'], dim=-1)

        score_tensor[i*batch_size:(i+1)*batch_size] = logits
    return score_tensor

def shift_left(tensor, pad_token = 50256):
    _, idx = torch.sort((tensor == pad_token), dim=1, stable=True)
    return torch.gather(tensor, -1, idx)
        
        
#response= query_model(batch.to('cuda'), gpt2_model, 1, 32)
#scores = get_scores(response.to('cuda'), value_model, 1, 32)