from __future__ import unicode_literals, print_function, division
from common import DataList, Groups

import math
import os
import sys
from six import string_types
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from contextlib import redirect_stderr
with redirect_stderr(open(os.devnull, "w")):
  import keras.models
import numpy as np
from magpie.base.document import Document
from magpie.base.word2vec import train_word2vec, fit_scaler
from magpie.config import NN_ARCHITECTURE, BATCH_SIZE, EMBEDDING_SIZE, EPOCHS
from magpie.nn.input_data import get_data_for_model
from magpie.nn.models import get_nn_model, root_mean_squared_error
from magpie.utils import save_to_disk, load_from_disk
from functools import reduce
import operator


def set_tf_growth():
  # tf.device("/cpu:0")
  a = 5
  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  # tf.keras.backend.set_session(tf.Session(config=config))


class Magpie(object):

  def __init__(self, keras_model=None, word2vec_model=None, scaler=None,
               labels=None):
    self.labels = labels

    if isinstance(keras_model, string_types):
      self.load_model(keras_model)
    else:
      self.keras_model = keras_model

    if isinstance(word2vec_model, string_types):
      self.load_word2vec_model(word2vec_model)
    else:
      self.word2vec_model = word2vec_model

    if isinstance(scaler, string_types):
      self.load_scaler(scaler)
    else:
      self.scaler = scaler

  def train(
          self,
          train_data: DataList,
          test_data: DataList,
          labels,
          callbacks=None,
          nn_model=NN_ARCHITECTURE,
          batch_size=BATCH_SIZE,
          test_ratio=0.0,
          epochs=EPOCHS,
          verbose=1):
    """
    Train the model on given data
    :param train_dir: directory with data files. Text files should end with
    '.txt' and corresponding files containing labels should end with '.lab'
    :param vocabulary: iterable containing all considered labels
    :param test_dir: directory with test files. They will be used to evaluate
    the model after every epoch of training.
    :param callbacks: objects passed to the Keras fit function as callbacks
    :param nn_model: string defining the NN architecture e.g. 'crnn'
    :param batch_size: size of one batch
    :param test_ratio: the ratio of samples that will be withheld from training
    and used for testing. This can be overridden by test_dir.
    :param epochs: number of epochs to train
    :param verbose: 0, 1 or 2. As in Keras.

    :return: History object
    """

    set_tf_growth()

    if not self.word2vec_model:
      raise RuntimeError('word2vec model is not trained. ' +
                         'Run train_word2vec() first.')

    if not self.scaler:
      raise RuntimeError('The scaler is not trained. ' +
                         'Run fit_scaler() first.')

    if self.keras_model:
      print('WARNING! Overwriting already trained Keras model.',
            file=sys.stderr)

    self.labels = labels
    self.keras_model = get_nn_model(
        nn_model,
        embedding=self.word2vec_model.vector_size,
        output_length=len(self.labels)
    )
    regression = nn_model == 'cnn_regression'  # TODO make this more general
    self.training_set = set([example['text'] for example in train_data])
    (x_train, y_train), test_data_matrix = get_data_for_model(train_data,
                                                              test_data,
                                                              self.labels,
                                                              nn_model=self.keras_model,
                                                              as_generator=False,
                                                              batch_size=batch_size,
                                                              word2vec_model=self.word2vec_model,
                                                              scaler=self.scaler,
                                                              regression=regression
                                                              )

    return self.keras_model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=test_data_matrix,
        # TODO make validation data optional for speedup
        callbacks=callbacks or [],
        verbose=verbose,
    )

  def predict_from_file(self, filepath):
    """
    Predict labels for a txt file
    :param filepath: path to the file

    :return: list of labels with corresponding confidence intervals
    """
    doc = Document(0, filepath)
    return self._predict(doc)

  def predict_from_text(self, text, test=False, return_float=False):
    """
    Predict labels for a given string of text
    :param text: string or unicode with the text
    :return: list of labels with corresponding confidence intervals
    """
    if hasattr(self, 'training_set') and not test:
      if text in self.training_set:
        print(f'found text in training set: {text}')
      # assert(not test or (text not in self.training_set))
    # else:
    #   # print("pretrained model not checking for test train split")
    doc = Document(text)
    return self._predict(doc, return_float=return_float)

  def _predict(self, doc: Document, return_float=False):
    """
    Predict labels for a given Document object
    :param doc: Document object
    :return: list of labels with corresponding confidence intervals
    """
    set_tf_growth()
    if isinstance(self.keras_model.input, list):
      _, sample_length, embedding_size = self.keras_model.input_shape[0]
    else:
      _, sample_length, embedding_size = self.keras_model.input_shape
    words = doc.get_all_words()[:sample_length]
    x_matrix = np.zeros((1, sample_length, embedding_size))

    for i, w in enumerate(words):
      if w in self.word2vec_model.wv:
        word_vector = self.word2vec_model.wv[w].reshape(1, -1)
        scaled_vector = self.scaler.transform(word_vector, copy=True)[0]
        x_matrix[0][i] = scaled_vector

    if isinstance(self.keras_model.input, list):
      x = [x_matrix] * len(self.keras_model.input)
    else:
      x = [x_matrix]

    with tf.device('/cpu:0'):
      y_predicted = self.keras_model.predict(x)
    # return weighted avg of labels
    # return reduce(lambda acc, x: acc + (x[0] * x[1]), zipped, 1) #weighted avg
    # TODO make this return weighted avg or max prob a param
    # max probablitiy, corresponding to standard keras mmethodology
    # print(f'model output shape {self.keras_model.output_shape}')
    if self.keras_model.output_shape[1] == 1:
      # print(f'returning {y_predicted[0][0]}')
      float_y_pred = float(y_predicted[0][0])
      # if not isinstance(y_predicted[0][0], float):
      #   print(type(y_predicted[0][0]))
      #   print(y_predicted, y_predicted[0][0])
      assert(isinstance(float_y_pred, float))
      # print(float_y_pred)
      return float_y_pred
    elif return_float:
      zipped = zip(self.labels, y_predicted[0])
      return float(
          sorted(
              zipped,
              key=lambda elem: elem[1],
              reverse=True)[0][0])
    else:
      zipped = zip(self.labels, y_predicted[0])
      return sorted(zipped, key=lambda elem: elem[1], reverse=True)[0][0]

  def train_word2vec(self, train_dir, vec_dim=EMBEDDING_SIZE):  # TODO consider deleting
    """
    Train the word2vec model on a directory with text files.
    :param train_dir: directory with '.txt' files
    :param vec_dim: dimensionality of the word vectors

    :return: trained gensim model
    """
    if self.word2vec_model:
      print('WARNING! Overwriting already trained word2vec model.',
            file=sys.stderr)

    self.word2vec_model = train_word2vec(train_dir, vec_dim=vec_dim)

    return self.word2vec_model

  def fit_scaler(self, data: DataList):
    """
    Fit a scaler on given data. Word vectors must be trained already.
    :param data: directory with '.txt' files

    :return: fitted scaler object
    """
    if not self.word2vec_model:
      raise ValueError('word2vec model is not trained. ' +
                       'Run train_word2vec() first.')

    if self.scaler:
      print('WARNING! Overwriting already fitted scaler.',
            file=sys.stderr)

    self.scaler = fit_scaler(data, word2vec_model=self.word2vec_model)

    return self.scaler

  def save_scaler(self, filepath, overwrite=False):
    """ Save the scaler object to a file """
    if not self.scaler:
      raise ValueError("Can't save the scaler, " +
                       "it has not been trained yet")
    save_to_disk(filepath, self.scaler, overwrite=overwrite)

  def load_scaler(self, filepath):
    """ Load the scaler object from a file """
    self.scaler = load_from_disk(filepath)

  def save_word2vec_model(self, filepath, overwrite=False):
    """ Save the word2vec model to a file """
    if not self.word2vec_model:
      raise ValueError("Can't save the word2vec model, " +
                       "it has not been trained yet")
    save_to_disk(filepath, self.word2vec_model, overwrite=overwrite)

  def load_word2vec_model(self, filepath):
    """ Load the word2vec model from a file """
    self.word2vec_model = load_from_disk(filepath)

  def save_model(self, filepath):
    """ Save the keras NN model to a HDF5 file """
    if not self.keras_model:
      raise ValueError("Can't save the model, "
                       "it has not been trained yet")

    # if os.path.exists(filepath):
    #   raise ValueError("File " + filepath + " already exists!")
    self.keras_model.save(filepath)

  def load_model(self, filepath):
    """ Load the keras NN model from a HDF5 file """
    if not os.path.exists(filepath):
      raise ValueError("File " + filepath + " does not exist")
    self.keras_model = keras.models.load_model(
        filepath, custom_objects={
            'root_mean_squared_error': root_mean_squared_error})
