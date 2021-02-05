from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from nwp import NWP
import pickle
import numpy as np

predictor = NWP(int(open("num_vocabulary.txt","r").read()), saved_finetune='real_save/best_model')

vocabulary = pickle.load(open("vocabulary.pckl", "rb"))
vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))

def predict_next_word(test_phrases):
    test_classes = predictor.inference(test_phrases)
    top_k = tf.math.top_k(test_classes, k=10)
    top_k = top_k[1].numpy()[0]

    latest_word = test_phrases[0].split(' ')[-1]

    words = [vocabulary[top_k_elem] for top_k_elem in top_k if vocabulary[top_k_elem] != latest_word]
    return words

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/join', methods=['GET','POST'])
def my_form_post():
    search = request.form['text1']
    search = ' '.join(search.split(' ')[-6:])
    result = predict_next_word([search])
    print(result)
    results = {}
    for i in range(len(result)):
        results[str('output_'+str(i))] = result[i]

    print(results)

    result = {str(key): [value] for key, value in results.items()}
    print(result)
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=False)
