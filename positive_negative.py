import os
from tqdm import tqdm

positive_ids = set()

for line in open('positive.txt'):
    positive_ids.add(line.strip())

folder = '20180514'

f_positive = open('positive.data', 'wb')
f_negative = open('negative.data', 'wb')

for file_name in tqdm(os.listdir(folder)):
    if file_name.startswith('.'):
        continue
    for line in tqdm(open(os.path.join(folder, file_name), 'rb')):
        inner_id, rest = line.split('\t', 1)
        rest = rest.strip()
        if rest:
            if inner_id in positive_ids:
                f_positive.write(line)
            else:
                f_negative.write(line)