from tf_model import CnnMaxPool
import tensorflow as tf
model = CnnMaxPool()
tf.train.Saver().restore(model.sess, "data/save/model_2_256")
accuracy, recall, f1 = model.test(model.loader.test)
print(accuracy, recall, f1)