from __future__ import print_function, unicode_literals

import io
import os
import nltk
import string

from nltk.tokenize import WordPunctTokenizer, sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)  # make sure it's downloaded before using


class Document(object):
  """ Class representing a document that the keywords are extracted from """

  def __init__(self, text, otherFeatures=None):
    # self.doc_id = doc_id

    if text:
      self.text = text
      self.filename = None
      self.filepath = None

    self.wordset = self.compute_wordset()

  def __str__(self):
    return self.text

  def compute_wordset(self):
    tokens = WordPunctTokenizer().tokenize(self.text)
    lowercase = [t.lower() for t in tokens]
    return set(lowercase) - {',', '.', '!', ';', ':', '-', '', None}

  def get_all_words(self):
    """ Return all words tokenized, in lowercase and without punctuation """
    return [w.lower() for w in word_tokenize(self.text)
            if w not in string.punctuation]

  def read_sentences(self):
    lines = self.text.split('\n')
    raw = [sentence for inner_list in lines
           for sentence in sent_tokenize(inner_list)]
    return [[w.lower() for w in word_tokenize(
        s) if w not in string.punctuation] for s in raw]
