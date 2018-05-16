
vocabulary = set()

for line in open('positive.data'):
    inner_id, rest = line.split('\t', 1)
    words = rest.split()
    vocabulary.update(words)

for line in open('negative.data'):
    inner_id, rest = line.split('\t', 1)
    words = rest.split()
    vocabulary.update(words)

f = open('volcabulary.txt', 'w')
for w in vocabulary:
    f.write(w + '\n')
