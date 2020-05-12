from typing import *
import math
from random import Random
import random
from common import Groups, DataTupleList, Groups, Tuple, DataList
import pandas as pd


def pluck(input_dict, output_dict, num_of_files):
  for category, category_files in input_dict.items():
    input_dict[category] = category_files[:num_of_files]
    del input_dict[:num_of_files]


def fileDictToTupleList(groups: Groups) -> DataTupleList:
  files = []
  for category, category_files in groups.items():
    for filename in category_files:
      files.append((category, filename))
  return files


def addToFileDict(dictionary, key, item):
  if key in dictionary:
    dictionary[key].append(item)
  else:
    dictionary[key] = [item]


def feature_fill(data_list: DataTupleList):
  features = set()
  for ex in data_list:
    if isinstance(ex[1], list):
      for ex_single in ex[1]:
        features.update(ex_single.keys())
    else:
      features.update(ex[1].keys())
  generic_dict = {key: 0.0 for key in features}
  for i in range(len(data_list)):
    if isinstance(data_list[i][1], list):
      new_docset = []
      for ex in data_list[i][1]:
        new_dict = dict.copy(generic_dict)
        new_dict.update(ex)
        new_docset.append(new_dict)
        assert(len(new_dict.keys()) == len(features))
      data_list[i] = (data_list[i][0], new_docset)
    else:
      new_dict = dict.copy(generic_dict)
      new_dict.update(data_list[i][1])
      data_list[i] = (data_list[i][0], new_dict)
      assert(len(new_dict.keys()) == len(features))


def feature_fill_non_tuples(data_list: DataList, features: set):
  generic_dict = {key: 0.0 for key in features}
  for i in range(len(data_list)):
    if isinstance(data_list[i], list):
      dicts_to_fill = data_list[i]
    else:
      dicts_to_fill = [data_list[i]]
    new_dicts = []
    for j in range(len(dicts_to_fill)):
      new_dict = dict.copy(generic_dict)
      if i == 0 and j == 0:
        print(dicts_to_fill[j])
      new_dict.update(dicts_to_fill[j])
      new_dicts.append(new_dict)
      assert(len(new_dict.keys()) == len(features))
    if isinstance(data_list[i], list):
      data_list[i] = new_dicts
    else:
      data_list[i] = new_dicts[0]


def train_test_split(groups: Groups, random_module: type(
        Random), k, k_max) -> Tuple[Groups, Groups]:
  # Disallowing balancing here because it would hinder k-fold. Full dataset
  # balancing can be done at extraction time
  from runner import log  # TODO move log to its own module to avoid thsese local imports
  assert(k < k_max and k >= 0)
  files = fileDictToTupleList(groups)
  feature_fill(files)
  log('shuffling unbalanced data')
  # random.Random(2019).shuffle(files)
  df = pd.DataFrame({'a': files})
  df = df.sample(frac=1, random_state=2019)
  files = df['a'].tolist()
  test_file_count = math.floor(len(files) / k_max)
  log(k_max, test_file_count, len(files))
  split_start = test_file_count * k
  split_end = min(split_start + test_file_count, len(files))
  train_files = files[:split_start] + files[split_end:]
  test_files = files[split_start:split_end]
  log(split_start)
  log(split_end)
  print(f'{len(train_files)} train docsets')
  print(f'{len(test_files)} test docsets')
  test_files_dict, train_files_dict = {}, {}
  for category, filename in test_files:
    addToFileDict(test_files_dict, category, filename)
  for category, filename in train_files:
    addToFileDict(train_files_dict, category, filename)
  return train_files_dict, test_files_dict
