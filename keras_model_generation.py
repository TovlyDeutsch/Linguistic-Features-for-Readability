import os
from typing import *

import gensim
import numpy as np

from common import paramsToString, DataList, mkdir
from magpie.main import Magpie
import datetime
import tensorflow as tf
from contextlib import redirect_stderr
with redirect_stderr(open(os.devnull, "w")):
  import keras

tf.get_logger().setLevel('WARNING')


def genMagpie(train_data: DataList,
              word2vec_model: str,
              corpusName: str,
              params: Dict[str,
                           Any],
              config_name=None,
              labels=None, overwrite=False, model_dir='') -> Tuple[Magpie,
                                                                   bool]:
  """Returns a tuple of magpie object and bool that indicates whether the model is trained
  """
  modelName = model_dir
  print(modelName)
  print(f'model exists: {os.path.exists(modelName)}')
  scalerName = f'magpieSaves/{corpusName}_{config_name.replace("/", "-")}_scaler'
  print(scalerName)
  print(f'overwrite {overwrite}')
  mkdir('magpieSaves')
  if (not os.path.exists(
          modelName)) or (not os.path.exists(scalerName)) or overwrite:
    magpieObj = Magpie(word2vec_model=word2vec_model, labels=labels)
    print('fitting new scaler')
    magpieObj.fit_scaler(train_data)
    magpieObj.save_scaler(scalerName, overwrite=True)
    return (magpieObj, False)
  else:
    print('loading cached model')
    return (Magpie(
        word2vec_model=word2vec_model,
        scaler=scalerName, keras_model=modelName, labels=labels), True)


def genAndTrainMagpie(train_data: DataList,
                      test_data: DataList,
                      labels: List[str],
                      corpusName,
                      params: Dict[str,
                                   Any],
                      overwrite=False,
                      config_name=None) -> Tuple[type(Magpie),
                                                 Any]:
  word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
      'GoogleNews-vectors-negative300.bin', binary=True)
  run_name = f'{corpusName}_{config_name.replace("/", "-")}'
  magpieObj, trained = genMagpie(train_data, word2vec_model, corpusName, params, config_name=config_name,
                                 labels=labels, overwrite=overwrite, model_dir=f'magpieSaves/{run_name}.h5')
  history = None
  if not trained or overwrite:
    log_dir = f"logs/{run_name}/" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    history = magpieObj.train(
        train_data,
        test_data,
        labels,
        epochs=params['epochs'],
        nn_model=params['nn_model'],
        batch_size=64,
        callbacks=[tensorboard_callback])  # TODO make batch_size param
    magpieObj.save_model(
        f'magpieSaves/{run_name}.h5')
  return magpieObj, history

# def trainAndEvaluate(
#         corpusName,
#         params):
#   _train_dir, test_dir = process_files(corpusName, params)
#   labels = ['WRLevel2', 'WRLevel3', 'WRLevel4', 'BitGCSE', 'BitKS3']
#   magpieObj = genAndTrainMagpie(corpusName, params)
#   test_x, test_y = magpieObj.docs_to_data(test_dir, labels)
#   results = magpieObj.keras_model.evaluate(x=test_x, y=test_y)
#   metric_labels = magpieObj.keras_model.metrics_names
#   return results, metric_labels


def writeExperiment(hyperparams, results, metric_labels):
  labeled_results = [
      f'{metric_labels[i]}: {results[i]}' for i in range(
          len(results))]
  import csv
  with open('evaluations.csv', 'a+') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(hyperparams + labeled_results)


# def runExperimentCrossEval(corpusName,
#                            k,
#                            params):
#   all_results = None
#   metric_labels = None
#   for i in range(k):
#     params['random_seed'] = i
#     results, metric_labels = trainAndEvaluate(corpusName, params)
#     if all_results is None:
#       all_results = np.array([results])
#     else:
#       all_results = np.append(all_results, [results], axis=0)
#   writeExperiment(
#       params.values() +
#       [f'0-{k}'],
#       np.mean(
#           all_results,
#           axis=0).tolist(),
#       metric_labels)
