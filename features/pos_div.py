# # from jnius import autoclass
# from statistics import stdev, mean
# from features.feature import Feature
# from text_parser import parse_text, get_scores
# from typing import List
# from collections import Counter, OrderedDict
# from scipy.stats import entropy
# import math
# # ScoredObject = autoclass('edu.stanford.nlp.util.ScoredObject')
# # Tree = autoclass('edu.stanford.nlp.trees.Tree')


# wordTags = [
#     "CD",
#     "DT",
#     "EX",
#     "FW",
#     "IN",
#     "JJ",
#     "JJR",
#     "JJS",
#     "LS",
#     "MD",
#     "NN",
#     "NNS",
#     "NNP",
#     "NNPS",
#     "PDT",
#     "POS",
#     "PRP",
#     "PRP$",
#     "RB",
#     "RBR",
#     "RBS",
#     "RP",
#     "SYM",
#     "TO",
#     "UH",
#     "VB",
#     "VBD",
#     "VBG",
#     "VBN",
#     "VBP",
#     "VBZ",
#     "WDT",
#     "WP",
#     "WP$",
#     "WRB"]
# phraseTags = ["NP", "VP"]
# clauseTags = ['SBAR']
# allTags = set(wordTags + phraseTags + clauseTags)


# def pos_counter(tree: type(Tree)) -> Counter:
#   taggedWords = tree.object().taggedYield().toArray()
#   labels = list(map(lambda w: w.tag(), taggedWords))
#   return Counter(labels)

# # TODO make sens common type and/or class


# def get_dists(sens: List[List[type(ScoredObject)]]):
#   return map(lambda sen: pos_counter(sen[0].object()), sens)


# def get_doc_dist(sens: List[List[type(ScoredObject)]]):
#   dists = get_dists(sens)
#   doc_dist = Counter()
#   for c in dists:
#     doc_dist.update(c)
#   return doc_dist


# class POSDocDev(Feature):
#   """ POSDdev (d) is the standard deviation of the distribution of the POS counts for
# document d. """

#   def generateFeature(self, text: str) -> float:
#     sens: List[List[type(ScoredObject)]] = parse_text(text, self.params['k'])
#     doc_dist = get_doc_dist(sens)
#     return stdev(doc_dist)


# def dist_div(dist: Counter, doc_dist: Counter) -> float:
#   keys = doc_dist.keys()
#   for key in keys:
#     if key not in dist:
#       dist[key] = 0
#   en = entropy(dist.values(), doc_dist.values())
#   if math.isnan(en):
#     # TODO deal with empty sentences
#     return 0
#   return en


# class POSAvgSenToDocDiv(Feature):
#   """ (POSD_div ). Let P be the distribution of the POS counts for document d. Let Q be the distribution
# of the POS counts for the corpus. POSd iv (d) = DK L(P || Q) """

#   def generateFeature(self, text: str, sens=None) -> float:
#     # sens: List[List[type(ScoredObject)]] = parse_text(text, self.params['k'])
#     doc_dist = get_doc_dist(sens)
#     dists = get_dists(sens)
#     divs = map(lambda dist: dist_div(dist, doc_dist), dists)
#     return mean(divs)

# # TODO consider adding doc to corpus div
