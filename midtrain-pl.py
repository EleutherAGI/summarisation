import torch
import transformers
import argparse
from utils import TLDRDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pytorch_lightning as pl

class LM(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.model = transformers.GPT2LMHeadModel.from_pretrained(self.args.model)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
       
    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask)['logits']
        
    def training_step(self, batch, batch_idx):
        text, summary_length = batch

        inputs = self.tokenizer(text, 
                            padding=True, 
                            truncation=True, 
                            return_length = True, 
                            max_length = 512, 
                            return_tensors = 'pt').to(self.model.device)

        total_length = inputs.pop('length')

        lm_logits = self(**inputs)

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
        return self.criterion(shift_logits[mask], shift_labels[mask])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def setup(self, stage=None):
        self.train_dataset = TLDRDataset(self.args.data + 'tldr-filtered-train.json', self.tokenizer)
        self.test_dataset = TLDRDataset(self.args.data + 'tldr-filtered-test.json', self.tokenizer)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(            
            self.train_dataset, 
            batch_size=self.args.test_batch_size,
            num_workers=2,
            pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.args.test_batch_size,
            num_workers=2,
            pin_memory=True)

def main():

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

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model = LM(args)
    trainer = pl.Trainer(gpus=-1, accelerator='ddp', precision=16)
    trainer.fit(model)

if __name__ == '__main__':
    main()