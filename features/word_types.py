from features.feature import FeatureFullExample
from nltk.tokenize import word_tokenize
from collections import Counter
from common import FeatureDict


class WordTypes(FeatureFullExample):
  def generateFeatures(self, example: FeatureDict) -> FeatureDict:
    word_types = word_tokenize(example['text'])
    counts = Counter(word_types)
    prepended_key_dict = {}
    for key, value in counts.items():
      prepended_key_dict[f'word_type_{key}'] = float(value)
    return prepended_key_dict


class WordTypeProportions(FeatureFullExample):
  def generateFeatures(self, example: FeatureDict) -> FeatureDict:
    word_types = word_tokenize(example['text'])
    counts = Counter(word_types)
    prepended_key_dict = {}
    for key, value in counts.items():
      prepended_key_dict[f'word_type_{key}'] = float(value) / len(word_types)
    return prepended_key_dict
