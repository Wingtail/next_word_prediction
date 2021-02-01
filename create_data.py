from nltk.corpus import gutenberg
from nltk import word_tokenize, sent_tokenize
import string
from tqdm import tqdm
import pickle
import csv
import random

def preprocess(raw_text):
    raw_text = (raw_text.replace('\n', ' ').replace('\r', ' ').replace('\ufeff', ' ').replace('_','')).lower()
    return raw_text

def compile_data(raw_text, train_writer, vali_writer, vocabulary, vocab_id, vali_split=0.2):
    print("Processing...")
    sents = sent_tokenize(preprocess(raw_text))
    passage = []

    data_count = 0

    for sent in tqdm(sents):
        words = word_tokenize(sent)
        cleaned_sent = [word for word in words if all(c.isalnum() for c in word)]
        if len(cleaned_sent) > 1:
            for i in range(1,len(cleaned_sent)-1):
                curr_sent = cleaned_sent[:i]
                target_word = cleaned_sent[i]
                if target_word not in vocabulary:
                    vocabulary[target_word] = vocab_id
                    vocab_id += 1

                if random.random() > vali_split:
                    train_writer.writerow([' '.join(curr_sent), vocabulary[target_word]])
                else:
                    vali_writer.writerow([' '.join(curr_sent), vocabulary[target_word]])
                data_count += 1

    print("{} data written".format(data_count))
    print("Curr vocabulary count: ", vocab_id)

    return vocabulary, vocab_id, data_count

def get_gutenberg(train_writer, vali_writer, vocabulary, vocab_id):
    #Gutenberg corpus
    total_data_count = 0
    for fileid in gutenberg.fileids():
        print("Gutenberg fileid: ", fileid)
        vocabulary, vocab_id, data_count = compile_data(gutenberg.raw(fileid), train_writer, vali_writer, vocabulary, vocab_id)
        total_data_count += data_count
    return vocabulary, vocab_id, total_data_count

def main():
    print("...Compiling NLTK Gutenberg text...")

    vocabulary = {}
    vocab_id = 0
    total_data_count = 0
    with open("corpus_vali.csv","w") as csv_valid:
        with open("corpus_train.csv", "w") as csv_train:
            train_writer = csv.writer(csv_train)
            vali_writer = csv.writer(csv_valid)

            train_writer.writerow(['sentence', 'label'])
            vali_writer.writerow(['sentence', 'label'])

            vocabulary, vocab_id, tot_count = get_gutenberg(train_writer, vali_writer, vocabulary, vocab_id)
            total_data_count += tot_count


    assert len(vocabulary.keys()) == vocab_id

    print("Compiling complete")
    print("{} data written".format(total_data_count))
    print("{} words in vocabulary".format(vocab_id))

    with open("num_vocabulary.txt","w") as f:
        f.write(vocab_id)

    print("Saving vocabulary")
    pickle.dump(vocabulary, open("vocabulary.pckl", "wb"))


if __name__ == '__main__':
    main()

