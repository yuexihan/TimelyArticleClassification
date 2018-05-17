import random


class Loader(object):
    def __init__(self, sanity_check=False):
        self.sanity_check = sanity_check
        self.w2v = self.load_word_vector()
        self.f_p_train = open('data/positive.train', 'rb')
        self.f_p_validate = open('data/positive.validate', 'rb')
        self.f_p_test = open('data/positive.test', 'rb')
        self.f_n_train = open('data/negative.train', 'rb')
        self.f_n_validate = open('data/negative.validate', 'rb')
        self.f_n_test = open('data/negative.test', 'rb')
        self.queue_p = []
        self.queue_n = []
        self.default_vec = [0] * 100
        print 'finish initialize data loader'

    def load_word_vector(self):
        f = open('data/volcabulary.vec', 'rb')
        w2v = {}
        for line in f:
            line = line.split()
            assert len(line) == 101
            w2v[line[0]] = line[1:]
            if self.sanity_check:
                if len(w2v) > 10000:
                    break
        return w2v

    def next_batch(self):
        result = []

        if len(self.queue_p) < 4:
            for _ in xrange(400):
                line = self.f_p_train.readline()
                if not line:
                    self.f_p_train.seek(0)
                    line = self.f_p_train.readline()
                _, rest = line.split('\t', 1)
                words = rest.split()
                vectors = []
                for w in words:
                    if w in self.w2v:
                        vectors.append(self.w2v[w])
                    else:
                        vectors.append(self.default_vec)
                self.queue_p.append((vectors, [1]))
            random.shuffle(self.queue_p)
        result.extend(self.queue_p[:4])

        if len(self.queue_n) < 60:
            for _ in xrange(6000):
                line = self.f_n_train.readline()
                if not line:
                    self.f_n_train.seek(0)
                    line = self.f_n_train.readline()
                _, rest = line.split('\t', 1)
                words = rest.split()
                vectors = []
                for w in words:
                    if w in self.w2v:
                        vectors.append(self.w2v[w])
                    else:
                        vectors.append(self.default_vec)
                self.queue_n.append((vectors, [0]))
        result.extend(self.queue_n[:60])

        return result
