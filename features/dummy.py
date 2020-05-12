from features.feature import Feature
from runner import gen_runner, log
from magpie.main import Magpie

# This feature was created to ensure that the effects of adding one
# feature where do to the feature itself and not simply model interaction


class Dummy(Feature):

  def generateFeature(self, text: str, parse=None) -> float:
    return 1.0
