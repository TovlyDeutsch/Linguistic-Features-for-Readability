import os

from random import Random
import random
from functools import reduce
# os.environ['CLASSPATH'] = 'C:/Users/we890/seniorThesis/stanford-parser-full-2018-10-17/stanford-parser.jar;C:/Users/we890/seniorThesis/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'
# import jnius_config
import json
# from jnius import autoclass, cast
from typing import *
import operator

FeatureDict = Dict[str, Union[str, float]]
DocSet = List[FeatureDict]
CorpusExamples = Union[FeatureDict, DocSet]
DataList = List[CorpusExamples]
DataList_FeatureDicts = List[FeatureDict]


Groups = Dict[str, DataList]
Groups_FeatureDicts = Dict[str, List[FeatureDict]]
DataTupleList = List[Tuple[str, FeatureDict]]


# def getLp():
#   LexicalizedParser = autoclass(
#       "edu.stanford.nlp.parser.lexparser.LexicalizedParser")
#   parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
#   return LexicalizedParser.loadModel(parserModel)


def getCorpusJson():
  jfile = open('corpora/newCorpus.json', 'r')
  return json.load(jfile)


def getCorpusJsonByName(filename: str) -> List[str]:
  jfile = open(f'corpora/{filename}.json', 'r')
  return json.load(jfile)


def paramsToString(params: Dict[str, Any]):
  '-'.join(['_'.join(map(str, param)) for param in params.items()])


def mkdir(dirname):
  try:
    os.makedirs(dirname)
  except FileExistsError:
    pass


def mkdir_run(dirname):
  for i in range(100):
    try:
      name = dirname + f'_{i}'
      os.makedirs(dirname + f'_{i}')
      return name
    except FileExistsError:
      pass


def groups_to_list(groups: Groups, random_module: Random):
  from runner import log
  output = list(reduce(operator.concat, groups.values()))
  for el in output:
    if 'label' not in el:
      print(el)
    assert('label' in el)
  # log('shuffling groups')
  # random.Random(0).shuffle(output)
  return output


T = TypeVar('T')


def generisize_func(
        func: Callable[[FeatureDict], T]) -> Callable[[CorpusExamples], T]:
  def generic_func(corpus_example: CorpusExamples):
    if isinstance(corpus_example, List):
      for el in corpus_example:
        func(el)
    else:
      func(corpus_example)
  return generic_func
