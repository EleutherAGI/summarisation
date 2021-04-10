import torch
import transformers
import argparse
from utils import TLDRDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, length, summary_length):

        lm_logits = self.model(input_ids = input_ids, 
                          attention_mask = attention_mask)['logits']

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Create mask for only end tokens
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for i, (s, t) in enumerate(zip(summary_length, length)):
                mask[i][t - s - 1 : t - 1] = True 
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        mask = mask.view(-1)
        # Calculate loss for summary only
        return self.criterion(shift_logits[mask], shift_labels[mask])

def train(args, model, tokenizer, device, train_loader, test_loader, optimizer, rank):

    model.train()
    model.zero_grad()


    for batch_idx, (text, summary_length) in enumerate(train_loader):

        inputs = tokenizer(text, padding=True, truncation=True, return_length = True, max_length = 512, return_tensors = 'pt').to(rank)
        
        loss = model(**inputs, summary_length = summary_length)

        loss /= args.accumulation_steps
        loss.backward()

        if (batch_idx+1) % args.accumulation_steps == 0: 
            optimizer.step()
            model.zero_grad()     

            if args.dry_run:
                break
            
            if rank == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                print(loss.item())
                print(batch_idx / len(train_loader))
                #wandb.log({"train_loss": loss.item()})    
            
        #run number of test phases per train epoch
        if (batch_idx+1) % (len(train_loader) // args.test_phases) == 0:
            test(args, model, tokenizer, device, test_loader)           

def test(args, model, tokenizer, device, test_loader, rank):

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    for batch_idx, (text, summary_length) in enumerate(test_loader):

        inputs = tokenizer(text, padding=True, truncation=True, return_length = True, max_length = 512, return_tensors = 'pt').to(device)
        total_length = inputs.pop('length')

        with torch.no_grad():
            lm_logits = model(**inputs)['logits']

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        # Create mask for only end tokens
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for i, (s, t) in enumerate(zip(summary_length, total_length)):
                mask[i][t - s - 1 : t - 1] = True 
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        mask = mask.view(-1)
        # Calculate loss for summary only
        loss = criterion(shift_logits[mask], shift_labels[mask])
        wandb.log({"test_loss": loss.detach().cpu().item()})

def main(rank, world_size, args, use_cuda):

    #--------------- Setup -------------#
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    #-----------------------------------#

    torch.manual_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = TLDRDataset(args.data + 'tldr-filtered-train.json', tokenizer)
    test_dataset = TLDRDataset(args.data + 'tldr-filtered-test.json', tokenizer)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,            
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)    

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,            
        num_workers=0,
        pin_memory=True,
        sampler=test_sampler)

    model = ModelWrapper(args.model).to(rank)

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if rank == 0:
        wandb.init('World-Pointer-Summarization')
        wandb.config.batch_size = args.batch_size
        wandb.config.learning_rate = args.lr
        wandb.config.seed = args.seed
        wandb.config.model = args.model

    for epoch in range(1, args.epochs + 1):
        train(args, model, tokenizer, device, train_loader, test_loader, optimizer, rank)

        if args.dry_run:
            print('finished dry run')
            break
    
    cleanup()

if __name__ == '__main__':
    #Training settings
    parser = argparse.ArgumentParser(description='Fine-Tune causal language model')

    parser.add_argument('--model', type=str, default='distilgpt2',
                        help='model type to train (default: distilgpt2)')

    parser.add_argument('--data', type=str, default='data/',
                        help='storage location of tldr data')

    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='number of gradient accumulation steps')

    parser.add_argument('--test-phases', type=int, default=4,
                        help='number of test phases per train epoch (default: 4)')

    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')

    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
                        
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--gpus', type=int, default=8, metavar='G',
                        help='number of gpus (default: 8)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    world_size = args.gpus

    if torch.cuda.device_count() > 1:
        print("We have available ", torch.cuda.device_count(), "GPUs! but using ",world_size," GPUs")

    #########################################################
    mp.spawn(main, args=(world_size, args, use_cuda), nprocs=world_size, join=True)    
    #########################################################