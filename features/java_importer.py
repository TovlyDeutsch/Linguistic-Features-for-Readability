from common import FeatureDict
from features.feature import FeatureFullExample
import csv
from typing import *


class JavaImporter(FeatureFullExample):
  def __init__(self, params: Dict[str, Any]) -> None:
    super().__init__(params)
    self.include = params.get('include', None)

  def cache_lookup(self):
    if not hasattr(self, 'lookup') or self.lookup is None:
      file_to_features = {}
      reader: OrderedDict[str, Any] = csv.DictReader(
          open('sample10NewselaAndWeebit.csv'))
      for f in reader:
        # print(f)
        filename = f['filename']
        f.pop('filename')
        for key, value in f.items():
          try:
            if self.include is None or key in self.include:
              f[key] = float(value)
          except ValueError as e:
            print(f'key is {key}')
            print(f'key is {value}')
          except TypeError as e:
            print(value)
            print(key)
          # if (type(value) != 'float'):
          #   print(value)
          #   print(f)
          #   assert(type(value) == 'float')
        file_to_features[filename] = f
      self.lookup = file_to_features

  def generateFeatures(self, example: FeatureDict) -> FeatureDict:
    # add prefix exclusions via params
    self.cache_lookup()
    features = self.lookup[example['filepath'].replace("\\", "/")]
    # print(f'in gen {features}')
    return features
