import os
from datetime import date

online_positive = set()

for line in open('data/online_positive.txt', encoding='utf-8'):
    online_positive.add(line.strip())

for day in range(1, 13):
    day = date(2018, 6, day)
    day = day.strftime('%Y%m%d')
    with open(os.path.join('data/online_positive', day), 'w', encoding='utf-8') as f:
        for line in open(os.path.join('data/article_info', day), encoding='utf-8'):
            create_time, inner_unique_id, rest = line.split('\t', 2)
            if inner_unique_id in online_positive:
                f.write(line)
