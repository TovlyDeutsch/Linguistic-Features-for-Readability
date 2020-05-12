
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from collections import Counter, defaultdict
import pickle
import os


def generate_vocabulary(data, save_vocab, vocabulary_size):
  all_data = " ".join(data)
  dd = defaultdict(int)
  words = [word.lower() for sent in sent_tokenize(all_data)
           for word in word_tokenize(sent)]
  for word in words:
    dd[word] += 1

  print("Len vocab initial: ", len(dd))
  vocabulary = []
  for word, freq in dd.items():
    if freq >= 5:
      vocabulary.append(word)
  print("Len vocab final: ", len(vocabulary))
  dirname = os.path.dirname(save_vocab)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(save_vocab, 'wb') as f:
    print("Saving corpus")
    pickle.dump(vocabulary, f)
  return vocabulary


class MyDataset(Dataset):

  def __init__(
          self,
          df,
          dict_path,
          task,
          max_length_sentences=10,
          max_length_word=20,
          vocabulary_size=10000):
    super(MyDataset, self).__init__()

    texts, labels = df.text.values.tolist(), df.readability.values.tolist()
    texts = [text.lower() for text in texts]

    if task in ["weebit", 'newsela']:
      # TODO check if this subtraction is necessary
      labels = [label for label in labels]
    else:
      labels = [label for label in labels]
    # print(len(labels), labels)

    # print(len(texts),texts[:100])
    #print(len(labels), labels[:100])

    self.texts = texts
    self.labels = labels
    if os.path.isfile(dict_path):
      self.dict = pickle.load(open(dict_path, 'rb'))
    else:
      self.dict = generate_vocabulary(texts, dict_path, vocabulary_size)

    self.max_length_sentences = max_length_sentences
    self.max_length_word = max_length_word
    self.num_classes = len(set(self.labels))

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    label = self.labels[index]
    text = self.texts[index]
    document_encode = [
        [
            self.dict.index(word) if word in self.dict else -
            1 for word in word_tokenize(
                text=sentences)] for sentences in sent_tokenize(
            text=text)]
    for sentences in document_encode:
      if len(sentences) < self.max_length_word:
        extended_words = [-1 for _ in range(
            self.max_length_word - len(sentences))]
        sentences.extend(extended_words)

    if len(document_encode) < self.max_length_sentences:
      extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                            range(self.max_length_sentences - len(document_encode))]
      document_encode.extend(extended_sentences)

    document_encode = [sentences[:self.max_length_word]
                       for sentences in document_encode][:self.max_length_sentences]

    document_encode = np.stack(arrays=document_encode, axis=0)
    document_encode += 1

    return document_encode.astype(np.int64), label
