import tensorflow as tf
from options import options
from tf_data_loader import Loader
import time


class CnnMaxPool(object):
    def __init__(self):
        self.options = options
        self.loader = Loader(sanity_check=False)
        self.sess = tf.Session()
        self.embedding = tf.constant(self.loader.id2v, dtype=tf.float32)
        self.build_graph()

    def forward(self, inputs):
        # embedding_lookup
        inputs = tf.nn.embedding_lookup(self.embedding, inputs)

        # 一维卷积网络
        with tf.variable_scope('cnn') as scope:
            try:
                conv_W_1 = tf.get_variable(name='conv_W_1', shape=(2, self.options['word_dimension'], self.options['channel_1']))
                conv_b_1 = tf.get_variable(name='conv_b_1', shape=self.options['channel_1'], initializer=tf.zeros_initializer())
                conv_W_2 = tf.get_variable(name='conv_W_2', shape=(2, self.options['word_dimension'], self.options['channel_2']))
                conv_b_2 = tf.get_variable(name='conv_b_2', shape=self.options['channel_2'], initializer=tf.zeros_initializer())
                conv_W_3 = tf.get_variable(name='conv_W_3', shape=(2, self.options['word_dimension'], self.options['channel_3']))
                conv_b_3 = tf.get_variable(name='conv_b_3', shape=self.options['channel_3'], initializer=tf.zeros_initializer())
            except ValueError:
                scope.reuse_variables()
                conv_W_1 = tf.get_variable(name='conv_W_1', shape=(2, self.options['word_dimension'], self.options['channel_1']))
                conv_b_1 = tf.get_variable(name='conv_b_1', shape=self.options['channel_1'], initializer=tf.zeros_initializer())
                conv_W_2 = tf.get_variable(name='conv_W_2', shape=(2, self.options['word_dimension'], self.options['channel_2']))
                conv_b_2 = tf.get_variable(name='conv_b_2', shape=self.options['channel_2'], initializer=tf.zeros_initializer())
                conv_W_3 = tf.get_variable(name='conv_W_3', shape=(2, self.options['word_dimension'], self.options['channel_3']))
                conv_b_3 = tf.get_variable(name='conv_b_3', shape=self.options['channel_3'], initializer=tf.zeros_initializer())

            conv_1 = tf.tanh(tf.nn.conv1d(inputs, conv_W_1, 1, 'SAME') + conv_b_1)
            conv_2 = tf.tanh(tf.nn.conv1d(inputs, conv_W_2, 1, 'SAME') + conv_b_2)
            conv_3 = tf.tanh(tf.nn.conv1d(inputs, conv_W_3, 1, 'SAME') + conv_b_3)

        pool_1 = tf.reduce_max(conv_1, 1)
        pool_2 = tf.reduce_max(conv_2, 1)
        pool_3 = tf.reduce_max(conv_3, 1)
        pool = tf.concat([pool_1, pool_2, pool_3], 1)

        # 全连接分类
        with tf.variable_scope('full') as scope:
            try:
                W = tf.get_variable(name='W', shape=(self.options['channel_1'] + self.options['channel_2'] + self.options['channel_3']))
                b = tf.get_variable(name='b', shape=(), initializer=tf.zeros_initializer())
            except ValueError:
                scope.reuse_variables()
                W = tf.get_variable(name='W', shape=(self.options['channel_1'] + self.options['channel_2'] + self.options['channel_3']))
                b = tf.get_variable(name='b', shape=(), initializer=tf.zeros_initializer())
            logits = tf.reduce_sum(tf.multiply(pool, W), 1) + b

        return logits

    def build_graph(self):
        self.inputs = tf.placeholder(tf.int32, (None, None))
        self.lens = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.float32, [None])

        self.logits = self.forward(self.inputs)
        self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.labels, self.logits, self.options['pos_weight']))
        losses = tf.get_collection('losses')
        if losses:
            self.loss += tf.add_n(losses)

        optimizer = tf.train.AdamOptimizer(self.options['learning_rate'])
        self.train_step = optimizer.minimize(self.loss)
        self.predictions = tf.cast(tf.greater(self.logits, 0), tf.float32)

    def train(self):
        sess = self.sess
        sess.run(tf.global_variables_initializer())
        last_time = time.time()
        for i in range(self.options['epoch']):
            for step in range(7297 // 4):
                inputs, lens, labels = self.loader.next_batch()
                feed_dict = {
                    self.inputs: inputs,
                    self.lens: lens,
                    self.labels: labels,
                }
                loss, _ = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
                if step % 200 == 0 and step != 0:
                    accuracy, recall, f1 = self.test(self.loader.validate)
                    now = time.time()
                    last_time, rate = now, 4*10/(now-last_time)
                    print('Step %6d: loss = %3.2f, accuracy = %2.3f, recall = %2.3f, f1 = %2.3f, docs/step = %8.2f' % (step, loss, accuracy, recall, f1, rate))
                print(step, loss)
        tf.train.Saver(sess, 'data/model')

    def test(self, data_set):
        sess = self.sess
        all_1 = 0       # 正样本总数
        get_1 = 0       # 判定为正样本的数目
        right_1 = 0     # 判定为正样本且正确的数目

        inputs = []
        lens = []
        labels = []
        for input, label in data_set:
            inputs.append(input)
            lens.append(len(input))
            labels.append(label)
            if len(inputs) >= 1024:
                print('hello')
                max_len = max(lens)
                self.loader.padding(inputs, max_len)
                feed_dict = {
                    self.inputs: inputs,
                    self.lens: lens,
                    self.labels: labels,
                }
                predictions = sess.run(self.predictions, feed_dict=feed_dict)
                for p, l in zip(predictions, labels):
                    if p == 1:
                        get_1 += 1
                        if l == 1:
                            right_1 += 1
                    if l == 1:
                        all_1 += 1
                inputs = []
                lens = []
                labels = []
        if len(inputs) > 0:
            max_len = max(lens)
            self.loader.padding(inputs, max_len)
            feed_dict = {
                self.inputs: inputs,
                self.lens: lens,
                self.labels: labels,
            }
            predictions = sess.run(self.predictions, feed_dict=feed_dict)
            for p, l in zip(predictions, labels):
                if p == 1:
                    get_1 += 1
                    if l == 1:
                        right_1 += 1
                if l == 1:
                    all_1 += 1
        if get_1 == 0:
            accuracy = right_1 / (get_1 + 0.1)
        else:
            accuracy = right_1 / get_1
        recall = right_1 / all_1
        if accuracy + recall == 0:
            f1 = 2 * accuracy * recall / (accuracy + recall + 0.1)
        else:
            f1 = 2 * accuracy * recall / (accuracy + recall)
        return accuracy, recall, f1


def main(_):
    model = CnnMaxPool()
    model.train()


if __name__ == '__main__':
    tf.app.run()
