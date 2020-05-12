"""
Readbility Assesment. Classes forked from work by Brian Yu
"""

from numpy.random import seed

from math import sqrt
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)  # TODO maybe make this a param

import logging
import os
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

from features.flesch import *
from features.feature import Feature
from magpie.main import Magpie
from sklearn.svm import SVC
import svm
from contextlib import redirect_stderr
with redirect_stderr(open(os.devnull, "w")):
  from keras import Model
  from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      print('setting mem growth')
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
from random import shuffle
import random
from common import groups_to_list, DocSet, FeatureDict, DataList
from dataSplitting import train_test_split, feature_fill, feature_fill_non_tuples
import ntpath
from common import DataList_FeatureDicts, Groups, Groups_FeatureDicts, generisize_func, mkdir, mkdir_run
import yaml
import numpy as np
from typing import *
from typing import cast
from os.path import isfile
from collections import Counter
import sys
import random
import sklearn
from keras_model_generation import genAndTrainMagpie
import json
import importlib
import argparse
from features import *
import pickle
import csv
import struct
import copy
from collections import defaultdict
try:
  import fcntl
except BaseException:
  print('running without fcntl')

  class FakeFcntl:
    def __init__(self):
      self.flock = lambda _a, _b: None
      self.LOCK_EX = 0
      self.LOCK_NB = 0
      self.LOCK_UN = 0
  fcntl = FakeFcntl()
import errno
import time

from sklearn import preprocessing
from statistics import mean, stdev

VERBOSE = False


def aquire_locked_file(filepath, mode='rb'):
  f = open(filepath, 'rb')
  while True:
    try:
      fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
      break
    except IOError as e:
      # raise on unrelated IOErrors
      if e.errno != errno.EAGAIN:
        raise
      else:
        time.sleep(0.1)
  return f


def safe_load(filepath: str):
  f = aquire_locked_file(filepath)
  return_val = pickle.load(f)
  fcntl.flock(f, fcntl.LOCK_UN)
  f.close()
  return return_val


def safe_dump(value, filepath: str):
  fr = None
  if os.path.isfile(filepath):
    fr = aquire_locked_file(filepath)
  fw = open(filepath, 'wb')
  if fr is None:
    fcntl.flock(fw, fcntl.LOCK_EX)
  pickle.dump(value, fw)
  if fr is None:
    fcntl.flock(fw, fcntl.LOCK_UN)
  fw.close()
  if fr is not None:
    fcntl.flock(fr, fcntl.LOCK_UN)
    fr.close()


def get_feature_class(feature_name: str) -> Type[Feature]:
  # feature imports are here to prevent circular dependencies (i.e. allow
  # feature modules to import from this file)
  from features.neural import Neural
  from features.dummy import Dummy
  from features.word_types import WordTypes
  from features.word_types import WordTypeProportions
  # from features.parser_uncertainty import PD, PDM
  # from features.pos_div import POSAvgSenToDocDiv, POSDocDev
  from features.java_importer import JavaImporter
  feature_class = eval(feature_name)
  assert(issubclass(feature_class, Feature))
  return feature_class


def log(message, override=False, clearline=False):
  if override or VERBOSE:
    if clearline:
      print(message, end="\r")
    else:
      print(message)


class Runner():

  def __init__(self, config):
    log("Initializing runner")
    self.config = config
    self.k = 0
    self.k_fold = config.k_fold
    self.folder = None
    self.regression = config.model.get('regression', False)
    # TODO set random seed here from config

  # TODO add runCrossEval
  def run_cross_eval(self, subsample_splits=1):
    import gc
    global VERBOSE
    # VERBOSE = False  # TODO remove
    assert(self.k_fold > 1)
    if self.config.fixed_k is not None:
      assert(self.config.fixed_k >= 0 and self.config.fixed_k < self.k_fold)
      k_range = range(self.config.fixed_k, self.config.fixed_k + 1)
    else:
      k_range = range(self.k_fold)
    self.load_data()
    self.weighted_f1s = {}
    self.macro_f1s = {}
    self.micro_f1s = {}
    self.rsmes = {}
    for i in k_range:
      log(f'Evaluating with k={i}')
      self.k = i
      if self.config.fixed_sample is not None:
        assert(self.config.fixed_sample >
               0 and self.config.fixed_sample <= subsample_splits)
        sample_range = range(
            self.config.fixed_sample,
            self.config.fixed_sample + 1)
        print(f'self.config.fixed_sample: {self.config.fixed_sample}')
      else:
        sample_range = range(1, subsample_splits + 1)
      for subsample in sample_range:
        self.generate_split(subsample=(subsample / subsample_splits))
        if len(self.config.features) > 0:
          log('generating features')
          self.generate_features()
        clf = self.train(subsample=subsample)
        self.evaluate(subsample=subsample)
        if not self.regression:
          self.weighted_f1s.setdefault(subsample, []).append(self.weighted_f1)
          self.macro_f1s.setdefault(subsample, []).append(self.macro_f1)
          self.micro_f1s.setdefault(subsample, []).append(self.micro_f1)
        else:
          self.rsmes.setdefault(subsample, []).append(self.rsme)
    import datetime
    now = datetime.datetime.now()
    if self.config.fixed_k is None:
      for subsample in range(1, subsample_splits + 1):
        if not self.regression:
          open(
              f'{self.folder}/{subsample}_avg_k_{self.k_fold}.txt',
              'w').write(
              f'Weighted F1: {str(mean(self.weighted_f1s[subsample]))}\nMacro F1: {str(mean(self.macro_f1s[subsample]))}'
              f'\nWeighted F1 std dev: {str(stdev(self.weighted_f1s[subsample]))}\nMacro F1 std dev: {str(stdev(self.macro_f1s[subsample]))}')
        else:
          open(
              f'{self.folder}/{subsample}_avg_k_{self.k_fold}.txt',
              'w').write(f'RSME: {str(mean(self.rsmes[subsample]))}')

  def load_and_annotate_data(self):
    self.load_data()
    self.generate_features()

  def run(self, k: int):
    self.k = k
    self.load_data()
    if len(self.config.features) > 0:
      log('generating features')
      self.generate_features()
    self.generate_split()
    clf = self.train()
    self.evaluate()
    return clf

  def get_trained_model(self, training_list: DataList, train_groups):
    self.training_list = training_list
    self.training_groups = train_groups
    self.test_list = copy.deepcopy(self.training_list)
    clf = self.train()
    return clf

  def load_data(self):
    log('Loading Data')
    self.data = self.getGroups(self.config.corpus)
    for group in self.data:
      print(f"{group} has {len(self.data[group])} docs")

    # TODO add shuffling here
    # TODO add saving of shuffled datasets

  def generate_features(self):
    from features.neural import Neural
    from features.java_importer import JavaImporter
    log('Generating Features')
    for group, examples in self.data.items():
      assert(None not in examples)
    for feature in self.config.features:
      log(feature)
      if isinstance(feature, Neural):
        feature.train(self.training_list, self.training_groups)
      feature.addFeatureToListDataset(self.training_list)
      feature.addFeatureToListDataset(self.test_list)
      # TODO add correlation printing for all features
      # if isinstance(feature, JavaImporter):
      #   feature.printCorrelations()
      feature_set = set()
      # gather a set of all the features
      for el in self.training_list + self.test_list:
        save_keys = generisize_func(lambda x: feature_set.update(x.keys()))
        save_keys(el)
      feature_fill_non_tuples(self.training_list, feature_set)
      feature_fill_non_tuples(self.test_list, feature_set)
      # for el in self.training_list + self.test_list:
      #   assert(set(el.keys()) == feature_set)

      # print(list(self.data.values())[0][0])
    mkdir('corpora/featureAnnotatedCorpora')
    safe_dump(
        self.data,
        f'corpora/featureAnnotatedCorpora/{self.config.corpus["name"]}_{self.config.name.replace("/", "-")}.pickle')

  @staticmethod
  def dict_values_list(
          docset_groups: Groups):
    for group, docset_list in docset_groups.items():
      docs_for_group = []
      for docset in docset_list:
        docs_for_group += docset
      docset_groups[group] = docs_for_group

  @staticmethod
  def seperate_docsets(data: Groups_FeatureDicts):
    slug_groups = defaultdict(list)
    for group, data_list in data.items():
      for feature_dict in data_list:
        slug_groups[feature_dict['slug']].append(feature_dict)
    print(f'len is {len(next(iter(slug_groups.values())))}')
    return {'genericGroup': [docset for docset in slug_groups.values()]}

  def generate_split(self, subsample=1.0):
    log(f'Splitting Data with k={self.k} and k_fold={self.k_fold}')
    docsets = False
    if self.config.corpus['name'] and self.config.corpus.get('docset', False):
      # assert(isinstance(self.data, Groups_FeatureDicts))
      docsets = True
      self.data = self.seperate_docsets(cast(Groups_FeatureDicts, self.data))
      # print(next(iter(self.data.values()))[0])
      # print(self.data['genericGroup'][0])
    self.training_groups, self.test_groups = train_test_split(
        self.data, self.config.random_module, self.k, self.k_fold)
    if docsets:
      print('converting docsets to docs')
      # print(next(iter(self.training_groups.values()))[0])
      # print(next(iter(self.training_groups.values()))[0])
      self.dict_values_list(self.training_groups)
      self.dict_values_list(self.test_groups)
      # print(next(iter(self.training_groups.values()))[0])
    self.training_list, self.test_list = groups_to_list(
        self.training_groups, self.config.random_module), groups_to_list(
        self.test_groups, self.config.random_module)
    print(subsample)
    num_training_docs = subsample * len(self.training_list)
    print(f'num docs before {len(self.training_list)}')
    print(int(num_training_docs))
    self.training_list = random.Random(
        num_training_docs).sample(self.training_list, int(num_training_docs))
    log(f'training with {num_training_docs} docs')
    log(f'testing with {len(self.test_list)} docs')

  def train(self, subsample=1) -> Union[Magpie, SVC]:
    from itertools import chain
    log('Training')
    model_name = self.config.model['name']
    label_set = set()
    if hasattr(self, 'test_groups'):
      group_range = chain(self.training_groups.values(), self.test_groups.values())
    else:
      group_range = self.training_groups.values()
    for group in group_range:
      for doc in group:
        label_set.add(doc['label'])
    labels = list(label_set)
    log(f'labels {labels}')
    # print(f'le params {self.le.transform(self.le.classes_)}') # TODO save le
    # for label in labels:
    #   assert(label in self.le.transform(self.le.classes_))
    # TODO highpri add exclusion of features not specified
    if model_name == 'transformer':
      from bert_from_matej.run_classifier import main as run_transformer
      # assert(isinstance(self.training_list, DataList_FeatureDicts))
      # assert(isinstance(self.test_list, DataList_FeatureDicts))
      self.training_list = cast(DataList_FeatureDicts, self.training_list)
      self.test_list = cast(DataList_FeatureDicts, self.test_list)
      processed_training_list = [[doc['text'], doc['label']]
                                 for doc in self.training_list]
      processed_test_list = [[doc['text'], doc['label']]
                             for doc in self.test_list]
      # TODO make this work for other corpora
      task_name = 'weebit' if self.config.corpus['name'] == 'WeeBit' else 'newsela'
      clf, self.weighted_f1, self.macro_f1, self.micro_f1, self.rsme = run_transformer(
          (processed_training_list, processed_test_list), k_fold=self.k, subsample=subsample, overwrite=self.config.model['overwrite'], config=self.config, task_name=task_name)
      return clf
    if model_name == 'han':
      from han.train import main as run_han
      # assert(isinstance(self.training_list, DataList_FeatureDicts))
      # assert(isinstance(self.test_list, DataList_FeatureDicts))
      self.training_list = cast(DataList_FeatureDicts, self.training_list)
      self.test_list = cast(DataList_FeatureDicts, self.test_list)
      processed_training_list = [[doc['text'], doc['label']]
                                 for doc in self.training_list]
      processed_test_list = [[doc['text'], doc['label']]
                             for doc in self.test_list]
      task_name = 'weebit' if self.config.corpus['name'] == 'WeeBit' else 'newsela'
      clf, self.weighted_f1, self.macro_f1, self.micro_f1, self.rsme = run_han(
          task_name, (processed_training_list, processed_test_list), k_fold=self.k, subsample=subsample, overwrite=self.config.model['overwrite'], config=self.config)
      return clf
    elif model_name != 'cnn' and model_name != 'cnn_regression':
      clf, self.report, self.weighted_f1, self.macro_f1, self.micro_f1, self.rsme = svm.train(
          self.training_list, self.test_list, [
              feature.name for feature in self.config.features], model=model_name, add_tfidf=self.config.tfidf)
      return clf
    else:
      print(
          f'right before genandtrain train: {len(self.training_list)}, test: {len(self.test_list)}')
      magpieObj, self.history = genAndTrainMagpie(
          self.training_list,
          self.test_list,
          labels,
          f"{self.config.corpus['name']}_subsample_{subsample}",
          self.config.model['params'], overwrite=self.config.model['overwrite'], config_name=f'{self.config.name}_k_{self.k}_subsample_{subsample}')
      # assert(isinstance(self.test_list, DataList_FeatureDicts))
      self.test_list = cast(DataList_FeatureDicts, self.test_list)
      y_pred = list(
          map(
              lambda text: magpieObj.predict_from_text(text, test=True), [
                  x['text'] for x in self.test_list]))
      y_true = [x['label'] for x in self.test_list]
      y_pred = list(map(lambda x: int(round(x)), y_pred))
      self.rsme = sqrt(sklearn.metrics.mean_squared_error(
          y_true, y_pred))
      print(f'rsme is {self.rsme}')
      if not self.regression:
        self.report = sklearn.metrics.classification_report(
            y_true, y_pred)
        self.weighted_f1 = sklearn.metrics.f1_score(
            y_true, y_pred, average='weighted')
        self.macro_f1 = sklearn.metrics.f1_score(
            y_true, y_pred, average='macro')
        self.micro_f1 = sklearn.metrics.f1_score(
            y_true, y_pred, average='micro')
      return magpieObj

  def evaluate(self, subsample=1):
    log('Evaluating')
    mkdir('reports')
    # print(self.report)
    import datetime
    now = datetime.datetime.now()
    if not hasattr(self, 'folder') or self.folder is None:
      self.folder = f'pickles'
      mkdir(self.folder)
    # if not self.regression:
      # f = open(
      #     f'{self.folder}/{subsample}_k_{self.k}.txt', 'w')
    report_obj = {
        'w_f1': getattr(self, 'weighted_f1', -1e10),
        'macro_f1': getattr(self, 'macro_f1', -1e10),
        'micro_f1': getattr(self, 'micro_f1', -1e10),
        'rsme': getattr(self, 'rsme', -1e10),
        'current_k': self.k,
        'current_sample_split': subsample,
        'max_sample_split': self.config.subsample_splits,
        'max_k': self.k_fold,
        'config': self.config.name,
        'limit': self.config.corpus.get('limit', None),
        'corpus': self.config.corpus['name']}
    print(report_obj)
    safe_dump(
        report_obj,
        f'{self.folder}/{subsample}_k_{self.k}_{self.config.corpus["name"]}_{self.config.name_no_ext}.pickle')
    # f.write(
    #     self.report +
    #     f'\nWeighted F1: {self.weighted_f1}\nMacro F1: {self.macro_f1}')
    # if hasattr(self, 'test_true_and_pred'):
    #   with open(f'{self.folder}/_k_{self.k}_results.txt', 'w+') as f:
    #     print(self.test_true_and_pred)
    #     f.write(json.dumps(self.test_true_and_pred))

  def getGroups(self, corpus) -> Groups:
    limit_str = f"_{corpus.get('limit', 'None')}"
    corpusPickleFilename = f'corpora/processedCorpora/regression_{self.regression}{corpus["name"]}_{limit_str}{"_docset" if corpus.get("docset") else ""}.pickle'
    corpusJSONFilename = f'corpora/processedCorpora/{corpus["name"]}_{limit_str}{"_docset" if corpus.get("docset") else ""}.json'
    # TODO highpri, make common annotated corpus with exclusion down below
    annotated_corpus = f'corpora/featureAnnotatedCorpora/{corpus["name"]}_{self.config.name.replace("/", "-")}.pickle'
    if isfile(annotated_corpus) and corpus.get('reprocess', False) == False:
      log('Loading cached anottated corpus')
      return safe_load(annotated_corpus)
    elif isfile(corpusPickleFilename) and corpus.get('reprocess', False) == False:
      log('Loading cached processed corpus')
      return safe_load(corpusPickleFilename)
    else:
      class ExtractorInterface:
        @staticmethod
        def getGroups(limit: int) -> Any: pass
      extractor = cast(
          ExtractorInterface,
          importlib.import_module(f'{corpus["name"]}'))
      # TODO make these extractors classes instead of modules
      assert(hasattr(extractor, 'getGroups'))
      groups: Dict[str, Any] = extractor.getGroups(corpus.get('limit'))
      if not self.regression:
        if corpus["name"] == 'WeeBit':
          level_lookup = {
              'BitGCSE': 4,
              'BitKS3': 3,
              'WRLevel2': 0,
              'WRLevel3': 1,
              'WRLevel4': 2}
          print('set le in non regression')
          self.le = lambda x: level_lookup[x]
        elif corpus["name"] == 'Newsela':
          def zero_tester(x):
            if x == '0.0' or 0 == int(float(x)):
              print(f'found 0 in runner for label {x}')
            return int(float(x)) - 2
          self.le = zero_tester
      elif self.regression == 'age':
        if corpus["name"] == 'WeeBit':
          age_lookup = {
              'BitKS3': 12.5,
              'BitGCSE': 15,
              'WRLevel2': 7.5,
              'WRLevel3': 8.5,
              'WRLevel4': 9.5}
        elif corpus["name"] == 'Newsela':
          age_lookup = {
              '2.0': 7.0,
              '3.0': 8.0,
              '4.0': 9.0,
              '5.0': 10.0,
              '6.0': 11.0,
              '7.0': 12.0,
              '8.0': 13.0,
              '9.0': 14.0,
              '10.0': 15.0,
              '11.0': 16.0,
              '12.0': 17.0}
        self.le = lambda x: age_lookup[x]
      elif self.regression == 'ordered_classes':
        if corpus["name"] == 'WeeBit':
          level_lookup = {
              'BitGCSE': 4.0,
              'BitKS3': 3.0,
              'WRLevel2': 0.0,
              'WRLevel3': 1.0,
              'WRLevel4': 2.0}
          self.le = lambda x: level_lookup[x]
        elif corpus["name"] == 'Newsela':
          self.le = lambda x: float(x)
      encoded_groups = {}
      for group, els in groups.items():
        label = self.le(group)
        if corpus["name"] == 'Newsela':
          encoded_groups[label] = [
              {'text': el[0], 'label': label, 'filepath': el[2], 'slug': el[3]} for el in els]
        else:
          encoded_groups[label] = [
              {'text': el[0], 'label': label, 'filepath': el[2]} for el in els]
      safe_dump(encoded_groups, corpusPickleFilename)
      group_list = groups_to_list(encoded_groups, self.config.random_module)
      # if not self.regression:
      #   json.dump([{'text': el['text'], 'filepath': el['filepath']}
      #              for el in group_list], open(corpusJSONFilename, 'w+'))
      return encoded_groups


class Config():

  def __init__(
          self,
          contents,
          name,
          fixed_k: int,
          fixed_sample: int,
          slurm: int):
    import math
    self.config = contents
    self.corpus = self.config["corpus"]
    self.features: List[Feature] = self.get_features()
    self.model = self.config['model']
    self.subsample_splits = self.config['subsample_splits']
    self.k_fold = self.config['k_fold']
    self.name = name
    self.tfidf = self.config.get('tfidf', False)
    if slurm is not None:
      assert(slurm < self.k_fold * self.subsample_splits and slurm >= 0)
      self.fixed_k = math.floor(slurm / self.subsample_splits)
      self.fixed_sample = 1 + (slurm % self.subsample_splits)
    else:
      self.fixed_k = fixed_k
      self.fixed_sample = fixed_sample
    self.name_no_ext, _ = os.path.splitext(ntpath.basename(self.name))
    self.random_module = random.Random()
    # TODO make this param and add it to annotatated corpora names
    self.random_module.seed(0)

  def get_features(self):
    return [get_feature_class(feature['name'])(feature)
            for feature in self.config.get("features", [])]


def main():
  config = parse_config()
  runner = Runner(config)
  if config.k_fold is not None:
    # print(config.k_fold)
    runner.run_cross_eval(subsample_splits=config.subsample_splits)
  else:
    runner.run(0)


def gen_runner(config_filename: str) -> Runner:
  return Runner(gen_config(config_filename))


def gen_config(config_filename: str, fixed_k=None,
               fixed_sample=None, slurm=None) -> Config:
  contents = open(config_filename).read()
  data = yaml.load(contents)
  return Config(data, config_filename, fixed_k, fixed_sample, slurm)


def parse_config():
  global VERBOSE
  parser = argparse.ArgumentParser(
      description="Run readability assesment experiments")
  parser.add_argument("config", type=str)
  # TODO figure out how add args
  parser.add_argument('-k', '--k_fold', type=int, action='store',
                      help="selects k to run among k fold")
  parser.add_argument('-s', '--sample', type=int, action='store',
                      help="selects sample to run among k sample selections")
  parser.add_argument('-a', '--array_num', type=int, action='store',
                      help="slurm convenience flag")
  # parser.add_argument("k", type=int, required=False)
  # parser.add_argument("sample", type=int, required=False)
  # parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()
  if args.verbose:
    VERBOSE = True
  return gen_config(
      args.config,
      getattr(
          args, 'k_fold', None),
      getattr(
          args, 'sample', None), getattr(
          args, 'array_num', None))


if __name__ == "__main__":
  main()

# example commands
# py runner.py -v configs/WeeBit/CnnFeatSVM.yaml
