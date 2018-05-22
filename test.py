from tf_model import CnnMaxPool
import tensorflow as tf
model = CnnMaxPool()
tf.train.Saver().restore(model.sess, "data/save/model")
accuracy, recall, f1 = model.test(model.loader.test)
print(accuracy, recall, f1)

result = []
inputs = []
lens = []
labels = []
for input, label in model.loader.test:
    inputs.append(input)
    lens.append(len(input))
    labels.append(label)
    if len(inputs) >= 128:
        max_len = max(lens)
        model.loader.padding(inputs, max_len)
        feed_dict = {
            model.inputs: inputs,
            model.lens: lens,
            model.labels: labels,
        }
        predictions = model.sess.run(model.logits, feed_dict=feed_dict)
        result.extend(predictions)
        inputs = []
        lens = []
        labels = []
if len(inputs) > 0:
    max_len = max(lens)
    model.loader.padding(inputs, max_len)
    feed_dict = {
        model.inputs: inputs,
        model.lens: lens,
        model.labels: labels,
    }
    predictions = model.sess.run(model.logits, feed_dict=feed_dict)
    result.extend(predictions)

with open('predictions_2.txt') as f:
    for r in result:
        f.write(str(r) + '\n')
