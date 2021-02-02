import csv
from matplotlib import pyplot as plt

vocab_statistic = []

with open('corpus_train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader)
    for row in spamreader:
        vocab_statistic.append(int(row[1]))

plt.hist(vocab_statistic, bins=100)
plt.title("Vocabulary statistics")
plt.xlabel("vocab_id")
plt.ylabel("num occurences")
plt.show()

