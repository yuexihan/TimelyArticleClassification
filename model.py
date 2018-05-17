# coding=utf-8
# coding=utf8
import dynet as dy
import numpy as np
from data_loader import Loader

class CnnMaxPool(object):
    def __init__(self):
        self.model = dy.Model()
        self.options = {'channel_1': 512, 'channel_2': 512, 'channel_3': 512}
        self.params = self.init_params()
        self.trainer = dy.AdamTrainer(self.model, alpha=0.01)
        self.loader = Loader(sanity_check=True)

    def load(self, filename):
        self.model.load(filename)

    def save(self, filename):
        self.model.save(filename)

    def init_params(self):
        params = {}

        # cnn层参数
        params['conv_W_1'] = self.model.add_parameters((1, 4, 100, self.options['channel_1']))
        params['conv_b_1'] = self.model.add_parameters(self.options['channel_1'])
        params['conv_W_2'] = self.model.add_parameters((1, 8, 100, self.options['channel_2']))
        params['conv_b_2'] = self.model.add_parameters(self.options['channel_2'])
        params['conv_W_3'] = self.model.add_parameters((1, 12, 100, self.options['channel_3']))
        params['conv_b_3'] = self.model.add_parameters(self.options['channel_3'])

        # 输出层参数
        params['W'] = self.model.add_parameters(self.options['channel_1']+self.options['channel_2']+self.options['channel_3'])
        params['b'] = self.model.add_parameters(1)
        return params

    def build_graph(self, x):
        conv_W_1 = dy.parameter(self.params['conv_W_1'])
        conv_b_1 = dy.parameter(self.params['conv_b_1'])
        conv_W_2 = dy.parameter(self.params['conv_W_2'])
        conv_b_2 = dy.parameter(self.params['conv_b_2'])
        conv_W_3 = dy.parameter(self.params['conv_W_3'])
        conv_b_3 = dy.parameter(self.params['conv_b_3'])
        W = dy.parameter(self.params['W'])
        b = dy.parameter(self.params['b'])

        (n, d), _ = x.dim()
        x = dy.reshape(x, (1, n, d))

        # 一维卷积网络
        conv_1 = dy.tanh(dy.conv2d_bias(x, conv_W_1, conv_b_1, (1, 1), is_valid=False))
        conv_2 = dy.tanh(dy.conv2d_bias(x, conv_W_2, conv_b_2, (1, 1), is_valid=False))
        conv_3 = dy.tanh(dy.conv2d_bias(x, conv_W_3, conv_b_3, (1, 1), is_valid=False))

        pool_1 = dy.max_dim(dy.reshape(conv_1, (n, self.options['channel_1'])))
        pool_2 = dy.max_dim(dy.reshape(conv_2, (n, self.options['channel_2'])))
        pool_3 = dy.max_dim(dy.reshape(conv_3, (n, self.options['channel_3'])))

        # 全连接分类
        pool = dy.concatenate([pool_1, pool_2, pool_3], 0)
        logit = dy.dot_product(pool, W) + b
        return logit

    def backward(self, word_vectors, label):
        dy.renew_cg()
        x = dy.inputTensor(word_vectors)
        y = dy.inputTensor(label)
        logit = self.build_graph(x)

        # q表示对正样本的加权
        # 公式见https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        q = 15
        l = 1 + (q - 1) * y
        loss = (1 - y) * logit + l * (dy.log(1 + dy.exp(-dy.abs(logit))) + dy.rectify(-logit))
        res = loss.value()
        loss.backward()
        return res

    def train(self):
        epoch = 5
        for i in xrange(epoch):
            for j in xrange(7297 / 4):
                for input, label in self.loader.next_batch():
                    loss = self.backward(input, label)
                    if np.isnan(loss):
                        print 'somthing went wrong, loss is nan.'
                        return
                self.trainer.update()
                print j, loss


if __name__ == '__main__':
    model = CnnMaxPool()
    model.train()
