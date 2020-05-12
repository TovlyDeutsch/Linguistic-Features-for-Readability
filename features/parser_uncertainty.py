# from text_parser import parse_text, get_scores
# # from jnius import autoclass
# from features.feature import Feature
# from statistics import stdev, mean
# from typing import List
# from functools import partial
# # ScoredObject = autoclass('edu.stanford.nlp.util.ScoredObject')


# class PD(Feature):
#   """ Parse Deviation: The parse deviation (PDx (s)) of sentence s is the standard deviation of the
# distribution of the x most probable parse log probabilities for s. If s has less than x valid parses, the distribution
# is taken from all those valid parses. """

#   def generateFeature(self, text: str, parse) -> float:
#     parses = parse
#     # parses: List[List[type(ScoredObject)]] = parse_text(text, self.params['k'])

#     def pd_sentence(sentence: List[type(ScoredObject)]) -> float:
#       scores = get_scores(sentence)[:self.params['k']]
#       if len(scores) < 2:
#         return 0
#       else:
#         return float(stdev(scores))
#     return mean(map(pd_sentence, parses))


# class PDM(Feature):
#   """ PDMx (s) is the difference between the largest parse log probability and the mean of the log probabilities of the
# x most probable parses for a sentence s. If s has less than x valid parses, the mean is taken over all those valid
# parses.
#  """

#   def generateFeature(self, text: str, parse=None) -> float:
#     # parses: List[List[type(ScoredObject)]] = parse_text(text, self.params['k'])
#     parses = parse

#     def pdm_sentence(sentence: List[type(ScoredObject)]) -> float:
#       scores = get_scores(sentence)[:self.params['k']]
#       return scores[0] - mean(scores)
#     return mean(map(pdm_sentence, parses))
