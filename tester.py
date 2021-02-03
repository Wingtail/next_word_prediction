import tensorflow as tf
from nwp import NWP
import pickle
import numpy as np

predictor = NWP(
5042
)

predictor.load_weights("ckpt")

vocabulary = pickle.load(open("vocabulary.pckl", "rb"))
vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))


def test_cases(model, test_phrases, vocabulary):
    test_classes = model.inference(test_phrases)
    top_k = tf.math.top_k(test_classes, k=10)
    top_k_probs = tf.nn.softmax(top_k[0]).numpy()[0]
    top_k = top_k[1].numpy()[0]

    idx = np.random.choice(top_k, p=top_k_probs)

    print(vocabulary[top_k[0]])

while True:
    phrase = input("Input: ")
    print(test_cases(predictor, [phrase], vocabulary))

