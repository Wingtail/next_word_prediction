import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

from tensorflow.keras import layers

class NWP(tf.keras.Model):
    def __init__(self, n_classes):
        super(NWP, self).__init__()
        self.USE = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False, dtype=tf.string)
        # self.dense1 = layers.Dense(512*4, activation=tfa.activations.gelu, kernel_initializer='zeros', bias_initializer='zeros')
        # self.dropout = layers.Dropout(0.3)
        # self.dense2 = layers.Dense(512, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros')
        self.dense2 = layers.Dense(512, activation=tfa.activations.gelu)

        self.classes = layers.Dense(n_classes)

    def __call__(self, x, training=True):
        x = self.USE(x)

        # att = self.dense1(x)
        # att = self.dropout(att, training=training)
        # att = self.dense2(att)
        x = self.dense2(x)

        # x = x + att

        x = self.classes(x)
        return x

    def inference(self, x):
        x = self.USE(x)

        # att = self.dense1(x)
        # att = self.dropout(att, training=training)
        # att = self.dense2(att)
        x = self.dense2(x)

        # x = x + att

        x = self.classes(x)
        x = tf.nn.softmax(x)
        return x
