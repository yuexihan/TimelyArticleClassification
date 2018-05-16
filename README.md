## 从cgi上拉取时新文章id
download_timely.py

## 根据时新文章id将分词结果分成正负样本文件
positive_negative.py
13850 positive.data
11137 positive.txt
4155361 negative.data
10423 positive_no_dup.data
2780215 negative_no_dup.data

## 建立词汇表，计算词向量
volcabulary.py
cat volcabulary.txt | ../word2vec/fastText/fasttext print-word-vectors words_100.bin > volcabulary.vec
词汇表大小 3526749

## 将样本按照7:1:3的比例随机分成训练组、验证组、测试组
train_validate_test.py
7297 positive.train
1042 positive.validate
2084 positive.test
109340 negative.train
15620 negative.validate
31263 negative.test
