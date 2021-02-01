import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers

class NWP(tf.keras.Model):
    def __init__(self, n_classes):
        super(NWP, self).__init__()
        self.USE = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False, dtype=tf.string)
        self.dense1 = layers.Dense(512, activation=tf.nn.relu6)
        self.dense2 = layers.Dense(512, activation=tf.nn.relu6)
        self.classes = layers.Dense(n_classes)

    def __call__(self, x):
        x = self.USE(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classes(x)
        return x

    def inference(self, x):
        x = self.USE(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classes(x)
        x = tf.nn.softmax(x)
        return x
