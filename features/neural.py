from features.feature import Feature
from runner import gen_runner, log
from magpie.main import Magpie


class Neural(Feature):
  def __init__(self, params):
    super().__init__(params)
    self.runner = gen_runner(self.params['model_config'])

  def train(self, train_list, train_groups):
    log('Generating neural model')
    self.runner = gen_runner(self.params['model_config'])
    self.predictor: Magpie = self.runner.get_trained_model(
        train_list, train_groups)

  def generateFeature(self, text: str, test=False) -> float:
    return self.predictor.predict_from_text(text, return_float=True)
