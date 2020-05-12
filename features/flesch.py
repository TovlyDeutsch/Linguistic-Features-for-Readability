# Modfied from code by Kwan-Yuet (Stephen) Ho:
# https://datawarrior.wordpress.com/2016/03/29/flesch-kincaid-readability-measure/
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
import nltk
# from common import getCorpusJson
from features.feature import Feature
nltk.download('cmudict', quiet=True)
nltk.download('punkt', quiet=True)


class FleschKincaid(Feature):
  def generateFeature(self, text: str) -> float:
    word_count, sent_count, syllable_count = text_statistics(text)
    return fk_formula(word_count, sent_count, syllable_count)


class Flesch(Feature):
  def generateFeature(self, text: str) -> float:
    word_count, sent_count, syllable_count = text_statistics(text)
    return flesch_formula(word_count, sent_count, syllable_count)


class WordCount(Feature):
  def generateFeature(self, text: str) -> float:
    return float(get_word_count(text))


class SentenceCount(Feature):
  def generateFeature(self, text: str) -> float:
    return float(get_sent_count(text))


class SyllableCount(Feature):
  def generateFeature(self, text: str) -> float:
    return float(get_syllable_count(text))


def not_punctuation(w): return not (len(w) == 1 and (not w.isalpha()))


def get_word_count(text: str):
  return len(
      list(filter(not_punctuation, word_tokenize(text))))


def get_sent_count(text): return len(sent_tokenize(text))


prondict = cmudict.dict()


def numsyllables_pronlist(l): return len(
    list(filter(lambda s: (s[-1]).isdigit(), l)))


def numsyllables(word):
  try:
    return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
  except KeyError:
    return [0]


def text_statistics(text):
  word_count = get_word_count(text)
  sent_count = get_sent_count(text)
  syllable_count = sum(
      map(
          lambda w: max(
              numsyllables(w)),
          word_tokenize(text)))
  return word_count, sent_count, syllable_count


def get_syllable_count(text):
  return sum(
      map(
          lambda w: max(
              numsyllables(w)),
          word_tokenize(text)))


def flesch_formula(word_count, sent_count, syllable_count): return 206.835 - \
    1.015 * word_count / sent_count - 84.6 * syllable_count / word_count


def fk_formula(word_count, sent_count, syllable_count): return 0.39 * \
    word_count / sent_count + 11.8 * syllable_count / word_count - 15.59


def syllablesPerWord(text):
  word_count, sent_count, syllable_count = text_statistics(text)
  return syllable_count / word_count


def avgSenLength(text):
  word_count, sent_count, syllable_count = text_statistics(text)
  return word_count / sent_count


def wordCount(text):
  word_count, sent_count, syllable_count = text_statistics(text)
  return word_count


def sent_count(text):
  word_count, sent_count, syllable_count = text_statistics(text)
  return sent_count


def syllable_count(text):
  word_count, sent_count, syllable_count = text_statistics(text)
  return syllable_count


# def perSent(metrics):
#   corpusJson = getCorpusJson()
#   docContents = list(map(lambda doc: doc['contents'], corpusJson))
#   assert(len(docContents) == len(corpusJson))
#   assert(len(docContents) == len(metrics))
#   corpusJson = getCorpusJson()
#   return list(map(lambda x: metrics[x[0]] /
#                   sent_count(x[1]), enumerate(docContents)))
