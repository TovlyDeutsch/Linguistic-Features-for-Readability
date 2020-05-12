from typing import *
import csv
from collections import defaultdict
import os

# def getGroupsDocsets():


def getGroups(limit=None) -> Dict[str, List[List[Union[str, int]]]]:
  groups = defaultdict(list)
  metadata_csv = csv.reader(
      open(
          'corpora/rawCorpora/newsela/articles_metadata.csv',
          'r',
          encoding="utf8"))
  counter = 0
  for i, row in enumerate(metadata_csv):
    if i == 0: continue
    slug = row[0]
    language = row[1]
    grade_level = row[3]  # this is the label
    filename = row[5]
    if '0' == str(int(float(grade_level))) or grade_level == 0:
      print(f'found grade level of {grade_level} for {filename}')
    if language == 'en':
      path = os.path.join('corpora/rawCorpora/newsela/articles', filename)
      text = open(path, 'r').read()
      groups[grade_level].append([text, grade_level, path, slug])
      counter += 1
  print(f'Extracted {counter} documents from Newsela')
  return groups
