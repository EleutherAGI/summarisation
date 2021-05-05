import json
import os
from sklearn.model_selection import train_test_split

dirs = os.listdir('./data/comparisons')
print(dirs)
#Read only relevant comparisons
dirs = [di for di in dirs if 'batch' in di and 'cnn' not in di]

data = []
for di in dirs:
    with open(f'./data/comparisons/{di}') as f:
        for line in f:
            data.append(json.loads(line))

train, test = train_test_split(data, test_size = .05)

with open('./data/comparisons-test.json', 'w') as outfile:
    json.dump({'data': test}, outfile)
    
with open('./data/comparisons-train.json', 'w') as outfile:
    json.dump({'data': train}, outfile)