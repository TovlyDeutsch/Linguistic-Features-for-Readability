from __future__ import unicode_literals, division
from common import DataList

import numpy as np
from gensim.models import Word2Vec

from magpie.base.document import Document
from magpie.config import BATCH_SIZE, SAMPLE_LENGTH
from magpie.utils import get_answers_for_doc, load_from_disk


def get_data_for_model(
        train_data: DataList,
        test_data: DataList,
        labels,
        nn_model=None,
        as_generator=False,
        batch_size=BATCH_SIZE,
        word2vec_model=None,
        scaler=None, regression=False):
  """
  Get data in the form of matrices or generators for both train and test sets.
  :param train_dir: directory with train files
  :param labels: an iterable of predefined labels (controlled vocabulary)
  :param test_dir: directory with test files
  :param nn_model: Keras model of the NN
  :param as_generator: flag whether to return a generator or in-memory matrix
  :param batch_size: integer, size of the batch
  :param word2vec_model: trained w2v gensim model
  :param scaler: scaling object for X matrix normalisation e.g. StandardScaler

  :return: tuple with 2 elements for train and test data. Each element can be
  either a pair of matrices (X, y) or their generator
  """

  kwargs = dict(
      label_indices={lab: i for i, lab in enumerate(labels)},
      word2vec_model=word2vec_model,
      scaler=scaler,
      nn_model=nn_model,
      regression=regression
  )

  train_data_matrix = build_x_and_y(train_data, **kwargs)
  test_data_matrix = build_x_and_y(test_data, **kwargs)

  return train_data_matrix, test_data_matrix


def build_x_and_y(data: DataList, **kwargs):
  """
  Given file names and their directory, build (X, y) data matrices
  :param filenames: iterable of strings showing file ids (no extension)
  :param file_directory: path to a directory where those files lie
  :param kwargs: additional necessary data for matrix building e.g. scaler

  :return: a tuple (X, y)
  """
  label_indices = kwargs['label_indices']
  word2vec_model = kwargs['word2vec_model']
  scaler = kwargs['scaler']
  nn_model = kwargs['nn_model']
  regression = kwargs.get('regression', False)

  x_matrix = np.zeros(
      (len(data),
       SAMPLE_LENGTH,
       word2vec_model.vector_size))
  if regression:
    # print('YES REGRESSION')
    y_matrix = np.zeros((len(data), 1), dtype=np.float_)
    # print(y_matrix)
  else:
    # print('NOT REGRESSION')
    y_matrix = np.zeros((len(data), len(label_indices)), dtype=np.bool_)

  for doc_id, example in enumerate(data):
    doc = Document(example['text'])
    words = doc.get_all_words()[:SAMPLE_LENGTH]

    for i, w in enumerate(words):
      if w in word2vec_model.wv:
        word_vector = word2vec_model.wv[w].reshape(1, -1)
        x_matrix[doc_id][i] = scaler.transform(word_vector, copy=True)[0]

    labels = [example['label']]

    for lab in labels:
      if regression:
        y_matrix[doc_id] = float(lab)
      else:
        index = label_indices[lab]
        y_matrix[doc_id][index] = True

  if nn_model and isinstance(nn_model.input, list):
    return [x_matrix] * len(nn_model.input), y_matrix
  else:
    return [x_matrix], y_matrix
