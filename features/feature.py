from abc import ABC, abstractmethod
# from text_parser import parse_text
from common import DataList, Groups, FeatureDict
from typing import *
from multiprocessing import Pool
import functools
from collections import defaultdict
import scipy


class Feature(ABC):

  def __init__(self, params: Dict[str, Any]) -> None:
    self.name = params["name"]
    assert(type(self).__name__ == self.name)
    self.reprocess = params.get('reprocess', False)
    self.params = params

  # TODO convert text to Document (class) instance
  # TODO case where this takes in parse should really be subclass
  @abstractmethod
  def generateFeature(self, text: str) -> float:
    """ Takes text and return the value of that feature computed on it"""
    pass

  def safeGenerateFeature(self, input_param) -> float:
    """ Takes text and return the value of that feature computed on it"""
    feature = self.generateFeature(input_param)
    if feature is None:
      print(f'{self.name} generated None feature')
      feature = 0.
    return feature

  # def gen_if_reprocess_pair(self, example_parse_pair):
  #   example, parse = example_parse_pair
  #   assert(example is not None)
  #   new_example = dict.copy(example)
  #   if self.name not in example or self.reprocess:
  #     new_example[self.name] = self.safeGenerateFeature(
  #         example['text'], parse=parse)
  #   return new_example

  def gen_if_reprocess(self, example):
    assert(example is not None)
    new_example = dict.copy(example)
    if self.name not in example or self.reprocess:
      new_example[self.name] = self.safeGenerateFeature(example['text'])
    # new_example.pop('filepath')
    # for feature_name, feature_val in new_example.items():
    #   if (type(feature_val) != float):
    #     print(f'violationkey nonjava {feature_name}')
    #     print(f'violationval nonjava {feature_val}')
    return new_example

  # def examples_and_parses(self, examples, k):
  #   for example in examples:
  #     yield example, parse_text(example['text'], k)

  def addFeatureToListDataset(self, dataset: DataList) -> None:
    for i in range(len(dataset)):
      dataset[i] = self.gen_if_reprocess(dataset[i])

  def addFeatureToDataset(self, dataset: Groups) -> None:
    """ Takes a dataset of type Groups and for each example generates and adds the given feature"""
    # pool = Pool(1)
    for group, examples in dataset.items():
      # # if 'k' in self.params:
      # if False:
      #   dataset[group] = list(
      #       pool.imap(
      #           self.gen_if_reprocess_pair,
      #           self.examples_and_parses(examples, self.params['k'])))
      # else:
      # dataset[group] = list(
      #     pool.map(
      #         self.gen_if_reprocess,
      #         examples))
      dataset[group] = list(
          map(
              self.gen_if_reprocess,
              examples))
    # pool.close()
    # TODO consider making this return modified instead of just modifying
  # TODO add calcCorrelation here


class FeatureFullExample(Feature):
  def __init__(self, params: Dict[str, Any]) -> None:
    super().__init__(params)
    self.feature_label_seqs: Dict[str, Tuple[List[float], List[float]]] = defaultdict(
        lambda: ([], []))

  def generateFeature(self, example: Dict[str, Any]) -> None:
    """ Takes text and return the value of that feature computed on it"""
    pass

  @abstractmethod
  def generateFeatures(self, example: Dict[str, Any]) -> FeatureDict:
    """ Takes text and return the value of that feature computed on it"""
    pass

  def printCorrelations(self):
    for feature, data in self.feature_label_seqs.items():
      # print(data)
      # TODO make this csv
      pearson = scipy.stats.pearsonr(data[0], data[1])
      spearman = scipy.stats.spearmanr(data[0], data[1])
      print(
          f"For {feature} pearson: ({pearson[0]:.4f}, {pearson[1]:.4f}) spearman: ({spearman[0]:.4f}, {spearman[1]:.4f})")

  def safeGenerateFeature(self, input_param) -> FeatureDict:
    """ Takes text and return the value of that feature computed on it"""
    features = self.generateFeatures(input_param)
    for feature_name, feature_val in features.items():
      if feature_val is None:
        print(f'{feature_name} generated None feature')
        features[feature_name] = 0.
      if isinstance(feature_val, float):
        self.feature_label_seqs[feature_name][0].append(feature_val)
        self.feature_label_seqs[feature_name][1].append(
            float(input_param['label']))
    return features

  def gen_if_reprocess(self, example):
    assert(example is not None)
    new_example = dict.copy(example)
    # print(f'copied dict {new_example}')
    if self.name not in example or self.reprocess:
      features = self.safeGenerateFeature(example)
      for feature_name, feature_val in features.items():
        # print(f'iteratting gen: {feature_name}, {feature_val}')
        # if (type(feature_val) != float):
        #   print(f'violationkey {feature_name}')
        #   print(f'violationkey {feature_val}')
        new_example[feature_name] = feature_val
      # print(new_example)
      # del new_example['filepath']
      # del new_example['label']
      # print(new_example)
    return new_example
