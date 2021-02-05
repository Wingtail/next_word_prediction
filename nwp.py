import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

from tensorflow.keras import layers

class NWP(tf.keras.Model):
    def __init__(self, n_classes):
        super(NWP, self).__init__()
        self.USE = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False, dtype=tf.string)
        self.dense2 = layers.Dense(512, activation=tfa.activations.gelu)

        self.classes = layers.Dense(n_classes)

    def __call__(self, x, training=True):
        x = self.USE(x)
        x = self.dense2(x)

        x = self.classes(x)
        return x

    def inference(self, x, training=False):
        x = self.USE(x, training=False)
        x = self.dense2(x)

        x = self.classes(x)
        x = tf.nn.softmax(x)
        return x

class NWPV2(tf.keras.Model):
    def __init__(self, n_classes):
        super(NWPV2, self).__init__()
        self.USE = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False, dtype=tf.string)
        self.dense1 = layers.Dense(512*4, activation=tfa.activations.gelu)
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(512)

        self.classes = layers.Dense(n_classes)
        # self.curr_temp = 1.0
        # self.temp_schedule = [1e-4, 1.0, 100000.0]

    # def update_temp(self, step):
        # self.curr_temp = min(self.temp_schedule[0] + (step * (self.temp_schedule[1]-self.temp_schedule[0])/self.temp_schedule[2]), self.temp_schedule[1])

    def __call__(self, x, training=True):
        x = self.USE(x, training=training)

        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)

        x = self.classes(x)
        return x

    def inference(self, x, training=False):
        x = self.USE(x, training=training)

        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)

        x = self.classes(x)
        x = tf.nn.softmax(x)
        return x
