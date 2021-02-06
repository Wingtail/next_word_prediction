# Next Word Prediction using Universal Sentence Encoder

A simple next word prediction model finetuned with Universal Sentence Encoder

======================

## Background
The project trains and performs next word prediction by finetuning the [**Universal Sentence Encoder**](https://tfhub.dev/google/universal-sentence-encoder/4)(USE). The deployed finetuned model can inference from roughly 25K unique vocabulary words.

In order to finetune the USE, I discovered the training data has to include not only the individual sentences in the text corpus, but also all the combinations of n_gram segments of each sentence. Ideally, the number of training data then approaches 2^n, where n is the number of words. 

## Implementation
USE was fintuned by adding a GELU activation non-linear layer, followed by a classification layer. I experimented many versions of different architectures, and the best performing model was the former architecture I stated.

USE was finetuned with a batch size of 64 examples using RectifiedAdam optimizer with learning rate 0.001 and weight decay of 1e-5 in a single GeForce RTX 2080 GPU for roughly 800K steps. I used RectifiedAdam to not worry about learning rate tuning. Indeed, RecitifiedAdam performed better than Adam, as loss curves for Adam suggest that the model had to repeatedly escape local minimas that prevented generalization. The model fully converged in 20K steps, and training further results in overfitting.

Implementing regularization techniques such as Dropout or L2 regularization on the Dense layers actually hurt the model's performance. This is understandable since we are finetuning a pretrained model. 

The green represents GELU non-linear layer trained in RectifiedAdam with lr 0.001, the pink represents a feed forward extension inspired by the Transformer model's "infinitely wide neural networks layer", and the orange represents Dropout of 0.2 followed by GELU non-linear layer trained in RectifiedAdam. 

<img src="https://github.com/Wingtail/next_word_prediction/blob/main/.github/images/train_log.png" data-canonical-src="![TTS banner](https://github.com/Wingtail/next_word_prediction/blob/main/.github/images/train_log.png =1080x640)
" width="1080" height="640" align="center" />

## Usage
### Installation
#### Deploy finetuned in webserver
```
1. Access the server directory
2. pip install -r requirements.txt
3. gunicorn --bind 0.0.0.0:8080 wsgi:application -w your_number_of_cpus --timeout=90
```
#### Install Development
**Do not install this if you just want to deploy the finetuned model in your own webserver.
The project is tested in python 3.8.5 and Tensorflow 2.2.
The project requires Tensorflow 2 and above and Tensorflow-Hub.
```
pip install -r requirements.txt
```
Installing NLTK corpus
Run your python interpreter, either with python or python3
In your interpreter, type
```
import nltk
nltk.download()
```
In the GUI, install all corpuses. If you do not want to install all corpuses, install only the Gutenberg and Webtext corpuses.

### Create Your Own Dataset
The dataset is custom generated. To reproduce the finetuned model, please run get_nltk_corpus.py and only include these text files:
```
1. grail.txt
2. pirates.txt
3. overheard.txt
4. melville-moby_dick.txt
5. wine.txt
```
You can also include austen's books in the text files as well (they bring interesting results). The Metamorphasis is a great text file as well. Please make sure to only include a small handful of correlated text data. A large corpora with uncorrelated text data does not perform well in my experience.

#### Standard Installation

```
1. Run get_nltk_corpus.py
You can also create a text_data directory and put in your own corpus. Make sure it is in .txt format and that you have done substantial text cleaning. The current dataset creator module does minimal text cleaning.

2. Run create_data.py
It will automatically create a vocabulary.pckl file and .csv files for training and validation using NWPDataCreator module. You do not have to manually index or shuffle data. The module does it automatically. The module also automatically removes too rate vocabulary words. Please understand that this module is built for USE finetuning. You can modify the module for implementing your own NLP pipeline.
```

### Finetune USE on Your Own
Make sure you installed and created your own dataset.
```
python train_nwp.py
```
You can also run tensorboard using the command
```
tensorboard --logdir ffw_log
```

