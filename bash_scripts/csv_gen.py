import os
import csv
from itertools import groupby
import pickle
from operator import itemgetter
from collections import defaultdict
from statistics import mean, stdev
import yaml
import sys


def get_pickle_groups():
  pickles = []
  for dirpath, dirnames, filenames in os.walk('pickles'):
    for filename in filenames:
      if filename.endswith('.pickle'):
        pickles.append(
            pickle.load(
                open(
                    os.path.join(
                        dirpath,
                        filename),
                    'rb')))
  grouped = defaultdict(list)
  for pickle_dict in pickles:
    grouped[pickle_dict['config']].append(pickle_dict)
  return grouped


def avg_metric(metric: str, dict_list):
  runs = []
  for run in dict_list:
    if metric not in run:
      return ''
    runs.append(run[metric])
  if(len(runs) != 5):
    print('non 5 runs found')
    return -1e10
  return mean(runs)


def std_metric(metric: str, dict_list):
  runs = []
  for run in dict_list:
    if metric not in run:
      return ''
    runs.append(run[metric])
  if(len(runs) != 5):
    print('non 5 runs found')
    return -1e10
  return stdev(runs)


def create_avg_csv(grouped, corpus_name: str):
  header = [
      'Features',
      'Weighted F1',
      'Macro F1',
      'Micro F1',
      'RMSE',
      'std dev weighted f1',
      'std dev macro f1',
      'std dev micro f1',
      'std dev rsme']
  f_600 = open(f'{corpus_name}_avg_600.csv', 'w', newline='')
  f = open(f'{corpus_name}_avg.csv', 'w', newline='')
  f_docsets = open(f'{corpus_name}_avg_docsets.csv', 'w', newline='')
  writer = csv.writer(f)
  writer_600 = csv.writer(f_600)
  writer_docset = csv.writer(f_docsets)
  writer_docset.writerow(header)
  writer.writerow(header)
  writer_600.writerow(header)
  corpus = pickle.load(
      open(
          f'corpora/processedCorpora/regression_False{corpus_name}__None.pickle',
          'rb'))
  length = sum([len(a) for a in corpus.values()])
  print(length)
  header = ['Features'] + [str(int((i / 10) * length))
                           for i in range(1, 10 + 1)]
  f_sub = open(f'{corpus_name}_macro_subsamples.csv', 'w', newline='')
  writer_sub = csv.writer(f_sub)
  writer_sub.writerow(header)
  for config, runs in grouped.items():
    final_runs = list(
        filter(
            lambda run: run['max_sample_split'] == run['current_sample_split'],
            runs))
    if len(final_runs) > 0:
      if os.path.isfile(config):
        yaml_file = yaml.load(open(config).read())
        if yaml_file['corpus']['name'] == corpus_name:
          if yaml_file['short_name'] == 'Flesch score + linear regression limit 600':
            count_mis = 0
            for run in final_runs:
              if 'rsme' not in run:
                count_mis += 1
            print(f'{count_mis} rmse in flesch linear')
            print(f'{len(final_runs)} runs for flesch linear')
          row = [yaml_file['short_name'],
                 avg_metric(
              'w_f1', final_runs), avg_metric(
              'macro_f1', final_runs), avg_metric(
              'micro_f1', final_runs), avg_metric(
              'rsme', final_runs), std_metric(
              'w_f1', final_runs), std_metric(
              'macro_f1', final_runs), std_metric(
              'micro_f1', final_runs), std_metric(
              'rsme', final_runs)]
          if 'docset' in config.lower():
            writer_docset.writerow(row)
          elif '600' in config:
            writer_600.writerow(row)
          else:
            writer.writerow(row)

    subsample_runs = list(
        filter(
            lambda run: run['max_sample_split'] == 10,
            runs))
    can_write = True
    if len(subsample_runs) > 0:
      if os.path.isfile(config):
        yaml_file = yaml.load(open(config).read())
        if yaml_file['corpus']['name'] == corpus_name:
          row = [yaml_file['short_name']]
          for subsample in range(1, 11):
            cur_subsample_runs = list(filter(
                lambda run: run['current_sample_split'] == subsample,
                subsample_runs))
            for k in range(5):
              can_write = can_write and (len(list(
                  filter(
                      lambda x: x['current_k'] == k,
                      cur_subsample_runs))) == 1)
              if not can_write:
                print(
                    f'cannot write b/c of {config} with subsample {subsample}')
                break
            if can_write:
              row.append(avg_metric('macro_f1', cur_subsample_runs))
          if can_write:
            print(f'able to write {row}')
            writer_sub.writerow(row)


def lookupShortName(config: str):
  return config


if __name__ == "__main__":
  groups = get_pickle_groups()
  create_avg_csv(groups, sys.argv[1])
