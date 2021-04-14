import json
from collections import OrderedDict, Counter
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data = []
with open('../../data/tldr/tldr-training-data.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

def view_sample(idx):
    print(f"content: {data[idx]['content']} \n")
    print(f"summary: {data[idx]['summary']} \n")
    print(f"subreddit: {data[idx]['subreddit']}")
    
view_sample(5)

len(data)

catagories = ['relationships',
            'AskReddit',
            'relationship_advice',
            'tifu',
            'dating_advice' ,
            'personalfinance',
            'Advice',
            'legaladvice',
            'offmychest',
            'loseit',
            'jobs',
            'self',
            'BreakUps',
            'askwomenadvice',
            'dogs',
            'running',
            'pettyrevenge',
            'needadvice',
            'travel',
            'Parenting',
            'weddingplanning',
            'Pets',
            'Dogtraining',
            'cats',
            'AskDocs',
            'college',
            'GetMotivated',
            'books',
            'Cooking']

#Whitelist catagories
whitelist = [item for item in data if 'subreddit' in item and item["subreddit"] in catagories]

#Remove duplicates
od = OrderedDict((hash(item['body']), item) for item in whitelist)
whitelist = list(od.values())

#Remove items whose body is longer than 512
#probably could just be done in training

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

trimmed = []
for item in tqdm(whitelist):
    if len(tokenizer(item['content'] + ' TLDR:' + item['summary'])['input_ids']) <= 512:
        trimmed.append(item)

subreddits = [item['subreddit'] for item in trimmed]
keys = list(Counter(subreddits).keys())
vals = list(Counter(subreddits).values())
tot = 0
for key, val in zip(keys, vals):
    tot += val
    print(f'{key}: {val}')
print(f'\ntotal items: {tot}')

train, test = train_test_split(trimmed, test_size = .05)

with open('../../data/tldr/tldr-filtered-test.json', 'w') as outfile:
    json.dump({'data': test}, outfile)
    
with open('../../data/tldr/tldr-filtered-train.json', 'w') as outfile:
    json.dump({'data': train}, outfile)

