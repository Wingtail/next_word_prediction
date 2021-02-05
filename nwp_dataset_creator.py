from nltk import word_tokenize, sent_tokenize
import string
from tqdm import tqdm
import pickle
import csv
import random
import os
import numpy as np

class NWPDataCreator:
    def __init__(self):
        self.vocabulary = {}

        self.vocab_id = 0

        self.freqs = []

        self.pos_weights = None

        self.total_data_count = 0

    def tokenize_dataset(self, article_dirs):
        print("------Tokenizing-----")

        '''
        Acquire global vocabulary frequency statistic
        '''
        for article_dir in article_dirs:
            print("Processing ", article_dir)
            with open(article_dir, "r") as f:
                raw_text = f.read()

            with open(os.path.splitext(article_dir)[0]+".csv", "w") as f:
                chunk_writer = csv.writer(f)
                self._compile_article_for_USE(raw_text, chunk_writer)
                print("Curr data count: ", self.total_data_count)
                print("Naive vocabulary count: ", self.vocab_id)

        self.freqs = sorted(self.freqs, key=lambda freq: freq[1], reverse=True)

        frequencies = np.array([freq[1] for freq in self.freqs])

        cum_frequencies = np.cumsum(frequencies)
        # freq_threshold = int(cum_frequencies[-1] * 0.9) #Account only 
        idx = np.amax(np.argwhere(frequencies >= 2))

        self.freqs = self.freqs[:idx]

        #Recompile vocabulary

        vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        self.vocabulary = {}
        self.vocab_id = 0

        #frequencies array lines up with vocab_id since freqs is sorted
        for vocab_id, _ in self.freqs:
            self.vocabulary[vocabulary[vocab_id]] = self.vocab_id
            self.vocab_id += 1

        print("Vocabulary in 10th percentile frequency processed!")
        print("Processed vocabulary count: ", self.vocab_id)

        #Calculate positive weights for Sparse Categorical Crossentropy
        frequencies = frequencies[:idx]
        print(frequencies)
        self.pos_weights = ((np.sum(frequencies)/frequencies) - 1.0)
        print("Pos weights: ")
        print(self.pos_weights)
        self.pos_weights /= self.pos_weights[0]
        self.pos_weights = np.log(self.pos_weights) + 1.0
        print(self.pos_weights)

    def _compile_article_for_USE(self, raw_text, writer):
        #Assume a single article fits in memory
        sents = sent_tokenize(self._preprocess(raw_text))
        passage = []

        data_count = 0

        for sent in tqdm(sents):
            words = word_tokenize(sent)
            cleaned_sent = [word for word in words if any(c.isalnum() for c in word)]
            if len(cleaned_sent) > 1:
                for n_gram in range(1, min(5, len(cleaned_sent))):
                    for i in range(max(len(cleaned_sent)-n_gram,1)):
                        curr_sent = cleaned_sent[i:i+n_gram]
                        target_word = cleaned_sent[i+n_gram]
                        if target_word not in self.vocabulary:
                            self.vocabulary[target_word] = self.vocab_id
                            self.freqs.append([self.vocab_id, 0])
                            self.vocab_id += 1

                        self.freqs[self.vocabulary[target_word]][1] += 1
                        passage.append((' '.join(curr_sent), target_word))
                        self.total_data_count += 1
                curr_sent = cleaned_sent[:-1]
                target_word = cleaned_sent[-1]
                if target_word not in self.vocabulary:
                    self.vocabulary[target_word] = self.vocab_id
                    self.freqs.append([self.vocab_id, 0])
                    self.vocab_id += 1

                self.freqs[self.vocabulary[target_word]][1] += 1
                passage.append((' '.join(curr_sent), target_word))
                self.total_data_count += 1

        random.shuffle(passage)

        for data_point in passage:
            writer.writerow(data_point)

    def _preprocess(self, raw_text):
        raw_text = (raw_text.replace('\n', ' ').replace('\r', ' ').replace('\ufeff', ' ').replace('_','')).lower()
        return raw_text

    def compile_data(self, article_dirs, vali_split=0.2):
        assert self.pos_weights is not None
        self.total_data_count = 0
        print("-----Compiling Data----")

        files = [open(os.path.splitext(article_dir)[0]+".csv", "r") for article_dir in article_dirs]
        csv_readers = [csv.reader(f) for f in files]

        with open("corpus_train.csv", "w") as trainf:
            with open("corpus_vali.csv", "w") as valif:
                train_writer = csv.writer(trainf)
                vali_writer = csv.writer(valif)

                train_writer.writerow(('sentence','label','pos_weight'))
                vali_writer.writerow(('sentence','label','pos_weight'))

                while len(files) > 0:
                    idx = random.randint(0,len(files)-1)

                    try:
                        data = next(csv_readers[idx])
                        if data[1] in self.vocabulary:
                            vocab_id = self.vocabulary[data[1]]
                            row = [data[0], vocab_id, self.pos_weights[vocab_id]]
                            if random.random() >= vali_split:
                                train_writer.writerow(row)
                            else:
                                vali_writer.writerow(row)
                            self.total_data_count += 1

                    except StopIteration:
                        files[idx].close()
                        files.pop(idx)
                        csv_readers.pop(idx)
                        continue

        print("Data Compiled!")
        print()
        print("--------Summary--------")
        print("Total data written: ", self.total_data_count)
        print("Processed vocabulary count: ", self.vocab_id)

        with open("num_vocabulary.txt", "w") as f:
            f.write(str(self.vocab_id))

    def save_vocab(self):
        print("----Saving Vocabulary-----")
        pickle.dump(self.vocabulary, open("vocabulary.pckl", "wb"))

    def create_data(self, article_dirs, vali_split=0.2):
        self.tokenize_dataset(article_dirs)
        self.compile_data(article_dirs, vali_split)
        self.save_vocab()
