import os
from datetime import date
from tqdm import tqdm

date_to_file = {}

folder = 'data/article_info'
for file_name in tqdm(os.listdir(folder)):
    if file_name.startswith('.'):
        continue
    for line in tqdm(open(os.path.join(folder, file_name), encoding='utf-8')):
        create_time, rest = line.split('\t', 1)
        if create_time:
            create_time = date.fromtimestamp(int(create_time))
            if create_time not in date_to_file:
                date_to_file[create_time] = open(os.path.join(folder, create_time.strftime('%Y%m%d')), 'w', encoding='utf-8')
            f = date_to_file[create_time]
            f.write(line)

for f in date_to_file.values():
    f.close()
