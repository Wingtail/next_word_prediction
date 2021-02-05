import tensorflow as tf
from nwp import NWP_old, NWP, FinetuneLayer, NWPV2
import pickle
import numpy as np

predictor = NWP_old(int(open("num_vocabulary.txt","r").read()))

predictor.load_weights("model_save/best_model")

transfer = FinetuneLayer(int(open("num_vocabulary.txt","r").read()))

predictor(["hello my name is"])
transfer(np.random.rand(1,512))

print(predictor.get_layer('dense_1').weights)
print(transfer.dense2.get_weights())

transfer.dense2.set_weights(predictor.dense2.get_weights())
transfer.classes.set_weights(predictor.classes.get_weights())

print(transfer.dense2.get_weights())

transfer.save_weights("real_save/best_model")
print("transfer complete")

# vocabulary = pickle.load(open("vocabulary.pckl", "rb"))
# vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))


# def test_cases(model, test_phrases, vocabulary):
#     test_classes = model.inference(test_phrases)
#     top_k = tf.math.top_k(test_classes, k=10)
#     top_k_probs = top_k[0].numpy()[0]
#     top_k_probs = top_k_probs*1.0/np.sum(top_k_probs)
#     top_k = top_k[1].numpy()[0]

#     idx = np.random.choice(top_k, p=top_k_probs)

#     splitted_phrase = test_phrases[0].split(' ')
#     print(vocabulary[idx])

#     # print([vocabulary[top_k[i]] for i in range(10) if vocabulary[top_k[i]] != splitted_phrase[-1]])

# while True:
#     phrase = input("Input: ")
#     print(test_cases(predictor, [phrase], vocabulary))

