import os
from datetime import date
from tqdm import tqdm

id_to_date = {}

for day in range(1, 13):
    day = date(2018, 6, day)
    day = day.strftime('%Y%m%d')
    for line in open(os.path.join('data/article_info', day), encoding='utf-8'):
        create_time, inner_unique_id, rest = line.split('\t', 2)
        id_to_date[inner_unique_id] = day

folder = 'data/article_tokens'
date_to_file = {}
for file_name in tqdm(os.listdir(folder)):
    if file_name.startswith('.') or file_name.startswith('2018'):
        continue
    for line in tqdm(open(os.path.join(folder, file_name), encoding='utf-8')):
        inner_id, rest = line.split('\t', 1)
        if inner_id in id_to_date:
            day = id_to_date[inner_id]
            if day not in date_to_file:
                date_to_file[day] = open(os.path.join(folder, day), 'w', encoding='utf-8')
            f = date_to_file[day]
            f.write(line)
