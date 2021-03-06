import random


class Loader(object):
    def __init__(self, sanity_check=False):
        self.sanity_check = sanity_check
        self.default_vec = [0] * 100
        self.w2v = self.load_word_vector()
        self.p_train = self.load_data('data/positive.train', 1)
        self.p_validate = self.load_data('data/positive.validate', 1)
        self.p_test = self.load_data('data/positive.test', 1)
        self.n_train = self.load_data('data/negative.train', 0)
        self.n_validate = self.load_data('data/negative.validate', 0)
        self.n_test = self.load_data('data/negative.test', 0)
        self.w2v = None
        self.p_i = 0
        self.n_i = 0
        print('finish initialize data loader')

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

    def load_data(self, file_name, label):
        f = open(file_name, 'rb')
        data = []
        for line in f:
            _, rest = line.split('\t', 1)
            words = rest.split()
            vectors = []
            for w in words:
                if w in self.w2v:
                    vectors.append(self.w2v[w])
                else:
                    vectors.append(self.default_vec)
            data.append((vectors, [label]))
        return data

    def next_batch(self):
        result = []

        for _ in range(4):
            if self.p_i == 0:
                random.shuffle(self.p_train)
            result.append(self.p_train[self.p_i])
            self.p_i = (self.p_i + 1) % len(self.p_train)

        for _ in range(60):
            if self.n_i == 0:
                random.shuffle(self.n_train)
            result.append(self.n_train[self.n_i])
            self.n_i = (self.n_i + 1) % len(self.n_train)

        return result
