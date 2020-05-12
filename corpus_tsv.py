import csv
import pickle
from common import groups_to_list
import random
corpus = groups_to_list(
    pickle.load(
        open(
            'corpora/processedCorpora/WeeBit__600.pickle',
            'rb')), random.Random(0))

keys = corpus[0].keys()
with open('corpus_train.csv', 'w', encoding='utf-8') as output_file:
  dict_writer = csv.DictWriter(output_file, keys)
  dict_writer.writerows(corpus)
