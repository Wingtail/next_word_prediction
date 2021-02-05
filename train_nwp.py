import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam
from tensorboard_logger import TensorboardLogger
from nwp import NWP, NWPV2
import numpy as np

import pickle
import os
from datetime import datetime

def get_dataset(file_path, batch_size):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=batch_size,
      label_name="label",
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
    return dataset

class Unpack(object):
    def __init__(self):
        pass
    def __call__(self, features, labels):
        return features['sentence'], labels, features['pos_weight']

def get_loss(model, x, labels, pos_weights):
    with tf.GradientTape() as tape:
        output = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, output, from_logits=True)
        loss = tf.math.reduce_mean(loss)
    return loss, tape.gradient(loss, model.trainable_variables), tf.nn.softmax(tf.stop_gradient(output))

def evaluate(model, x, labels, pos_weights):
    output = model(x, training=False)
    loss = tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, output, from_logits=True))
    return loss, output

def test_cases(model, test_phrases, vocabulary):
    test_classes = model.inference(test_phrases)
    top_k_words = tf.math.top_k(test_classes, k=5)[1].numpy()

    test_words = {}
    for i in range(len(top_k_words)):
        test_words[test_phrases[i]] = ', '.join([vocabulary[k_word] for k_word in top_k_words[i]])

    return test_words

def main():
    model = NWP(int(open("num_vocabulary.txt", "r").read()))
    epochs = 100
    global_step = 0
    log_time = 100

    test_phrases = [
        "i",
        "i am",
        "you are",
        "i was a little",
        "mixed precision",
        "you need",
        "there needs to",
        "i will send you the location",
        "take a look",
        "i am not too sure about that but you",
        "bob has to eat",
        "can you do me a",
        "it offers no math performance benefit for using the technique, but"
    ]

    vocabulary = pickle.load(open("vocabulary.pckl", "rb"))
    vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))

    logger = TensorboardLogger('./ffw_log/', 'ffw_model')

    optimizer = RectifiedAdam(0.001, weight_decay=0.000001)

    loss_accum = tf.keras.metrics.Mean()
    acc_accum = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)

    best_vali_loss = 1e12
    best_train_loss = 1e12
    checkpoint_dir = "./checkpoints/"+datetime.now().strftime("%B-%d-%Y_%I+%M%p")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        train_data = get_dataset("corpus_train.csv", 64).map(Unpack()).repeat(1)
        vali_data = get_dataset("corpus_vali.csv", 32).map(Unpack()).repeat(1)


        for text, labels, pos_weights in train_data:
            loss, grads, output = get_loss(model, text, labels, pos_weights)
            grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            acc_accum.update_state(labels.numpy(), output.numpy())
            loss_accum.update_state(loss.numpy())

            if global_step % log_time == 0:
                print("------- Training case ------- Global step ", global_step)
                stats = {
                    'loss':loss_accum.result().numpy(),
                    'accuracy':acc_accum.result().numpy()
                }
                print(stats)
                logger.dict_to_tb_scalar('TrainStats', stats, global_step)
                loss_accum.reset_states()
                acc_accum.reset_states()
                if stats['loss'] < best_train_loss:
                    best_train_loss = stats['loss']
                    model.save_weights(checkpoint_dir+"/best_train")

            if global_step % 10000 == 0:
                model.save_weights(checkpoint_dir+"/checkpoint_{}".format(global_step))


            global_step += 1


        loss_accum.reset_states()
        acc_accum.reset_states()

        print("Validation in Progress...")

        for text, labels, pos_weights in vali_data:
            loss, output = evaluate(model, text, labels, pos_weights)

            acc_accum.update_state(labels.numpy(), output.numpy())
            loss_accum.update_state(loss.numpy())

        stats = {
            'loss':loss_accum.result().numpy(),
            'accuracy':acc_accum.result().numpy()
        }

        if stats['loss'] <= best_vali_loss:
            best_vali_loss = stats['loss']
            model.save_weights(checkpoint_dir+"/best_model")

        print("---- Evaluation ------ Global Step ", global_step)
        print(stats)

        logger.dict_to_tb_scalar('ValidStats', stats, global_step)
        loss_accum.reset_states()
        acc_accum.reset_states()

        print("---- Test cases ------ Global Step ", global_step)
        test_stats = test_cases(model, test_phrases, vocabulary)

        print(test_stats)

        logger.dict_to_tb_text(test_stats, global_step)

if __name__=='__main__':
    main()
