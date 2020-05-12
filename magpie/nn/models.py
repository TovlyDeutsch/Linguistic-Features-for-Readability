from keras.layers import Input, Dense, GRU, Dropout, BatchNormalization, \
    MaxPooling1D, Conv1D, Flatten, Concatenate, LSTM, Bidirectional
from keras.models import Model, Sequential
import datetime
from magpie.config import SAMPLE_LENGTH
import tensorflow as tf
from contextlib import redirect_stderr
import os
with redirect_stderr(open(os.devnull, "w")):
  from keras import backend as K


def get_nn_model(nn_model, embedding, output_length):
  print('getting model')
  print(f'model name is {nn_model}')
  if nn_model == 'cnn':
    return cnn(embedding_size=embedding, output_length=output_length)
  if nn_model == 'cnn_regression':
    print('selecting regression cnn')
    return cnn(
        embedding_size=embedding,
        output_length=1,
        loss=root_mean_squared_error, metrics=[])
  elif nn_model == 'rnn':
    return rnn(embedding_size=embedding, output_length=output_length)
  elif nn_model == 'bilstm':
    return bilstm(embedding_size=embedding, output_length=output_length)
  else:
    raise ValueError("Unknown NN type: {}".format(nn_model))


def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true)))


def lstm(embedding_size, output_length):
  HIDDEN_LAYER_SIZE = 256

  inputs = Input(shape=(SAMPLE_LENGTH, embedding_size))

  gru = LSTM(
      HIDDEN_LAYER_SIZE,
      input_shape=(SAMPLE_LENGTH, embedding_size),
      kernel_initializer="glorot_uniform",
      recurrent_initializer='normal',
      activation='relu',
  )(inputs)

  batch_normalization = BatchNormalization()(gru)
  dropout = Dropout(0.1)(batch_normalization)
  outputs = Dense(output_length, activation='sigmoid')(
      dropout)  # try softmax also

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['categorical_accuracy'],
  )

  return model


def bilstm(embedding_size, output_length):
  HIDDEN_LAYER_SIZE = 256

  inputs = Input(shape=(SAMPLE_LENGTH, embedding_size))

  gru = Bidirectional(LSTM(
      HIDDEN_LAYER_SIZE,
      input_shape=(SAMPLE_LENGTH, embedding_size),
      kernel_initializer="glorot_uniform",
      recurrent_initializer='normal',
      activation='relu',
  ))(inputs)

  batch_normalization = BatchNormalization()(gru)
  dropout = Dropout(0.1)(batch_normalization)
  outputs = Dense(output_length, activation='sigmoid')(dropout)

  model = Model(inputs=inputs, outputs=outputs)
  print('compile lstm')
  model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['categorical_accuracy'],
  )

  return model


def cnn(
        embedding_size,
        output_length,
        loss='binary_crossentropy',
        metrics=['categorical_accuracy']):
  """ Create and return a keras model of a CNN """
  K.clear_session()  # TODO check if this is necessary

  NB_FILTER = 256
  NGRAM_LENGTHS = [1, 2, 3, 4, 5]

  conv_layers, inputs = [], []

  for ngram_length in NGRAM_LENGTHS:
    current_input = Input(shape=(SAMPLE_LENGTH, embedding_size))
    inputs.append(current_input)

    convolution = Conv1D(
        NB_FILTER,
        ngram_length,
        kernel_initializer='lecun_uniform',
        activation='tanh',
    )(current_input)

    pool_size = SAMPLE_LENGTH - ngram_length + 1
    pooling = MaxPooling1D(pool_size=pool_size)(convolution)
    conv_layers.append(pooling)

  merged = Concatenate()(conv_layers)
  dropout = Dropout(0.5)(merged)
  flattened = Flatten()(dropout)
  if output_length == 1:
    print('linear output')
    outputs = Dense(output_length, activation='linear')(flattened)
  else:
    outputs = Dense(output_length, activation='sigmoid')(flattened)

  model = Model(inputs=inputs, outputs=outputs)
  print('compile cnn test')
  print(f'output length {output_length}')

  model.compile(
      loss=loss,
      optimizer='adam',
      metrics=metrics,
  )

  return model


def rnn(embedding_size, output_length):
  """ Create and return a keras model of a RNN """
  HIDDEN_LAYER_SIZE = 256

  inputs = Input(shape=(SAMPLE_LENGTH, embedding_size))

  gru = GRU(
      HIDDEN_LAYER_SIZE,
      input_shape=(SAMPLE_LENGTH, embedding_size),
      kernel_initializer="glorot_uniform",
      recurrent_initializer='normal',
      activation='relu',
  )(inputs)

  batch_normalization = BatchNormalization()(gru)
  dropout = Dropout(0.1)(batch_normalization)
  outputs = Dense(output_length, activation='sigmoid')(dropout)

  model = Model(inputs=inputs, outputs=outputs)
  print('compile rnn')
  model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['categorical_accuracy'],
  )

  return model
