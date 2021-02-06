# next_word_prediction
A simple next word prediction model using Universal Sentence Encoder

Next Word Prediction using Universal Sentence Encoder
======================

# Background
The project trains and performs next word prediction by finetuning the [**Universal Sentence Encoder**](https://tfhub.dev/google/universal-sentence-encoder/4)(USE). The deployed finetuned model can inference from roughly 25K unique vocabulary words.

In order to finetune the USE, I discovered the training data has to include not only the individual sentences in the text corpus, but also all the combinations of n_gram segments of each sentence. Ideally, the number of training data then approaches 2^n, where n is the number of words. 

# Implementation
USE was fintuned by adding a GELU activation non-linear layer, followed by a classification layer. I experimented many versions of different architectures, and the best performing model was the former architecture I stated.

USE was finetuned with a batch size of 64 examples using RectifiedAdam optimizer with learning rate 0.001 and weight decay of 1e-5 in a single GeForce RTX 2080 GPU for roughly 800K steps. I used RectifiedAdam to not worry about learning rate tuning. Indeed, RecitifiedAdam performed better than Adam, as loss curves for Adam suggest that the model had to repeatedly escape local minimas that prevented generalization. The model fully converged in 20K steps, and training further results in overfitting.

Implementing regularization techniques such as Dropout or L2 regularization on the Dense layers actually hurt the model's performance. This is understandable since we are finetuning a pretrained model. 

The green represents GELU non-linear layer trained in RectifiedAdam with lr 0.001, the pink represents a feed forward extension inspired by the Transformer model's "infinitely wide neural networks layer", and the orange represents Dropout of 0.2 followed by GELU non-linear layer trained in RectifiedAdam. 

# 1. Usage
## Installation
**Do not install this if you just want to deploy the finetuned model in your own webserver.
The project is tested in python 3.8.5 and Tensorflow 2.2.
The project requires Tensorflow 2 and above and Tensorflow-Hub.
```
pip install -r requirements.txt
```
## Deploy finetuned in webserver
```
1. Access the server directory
2. pip install -r requirements.txt
```
