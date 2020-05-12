# This file was originally created by the authors of ___ and modified by TODO fill in
# Tovly Deutsch

import os
from bs4 import BeautifulSoup
import nltk
import langid
import pandas as pd
import string
import argparse
from typing import Dict, List, Union
import json


def get_BitGCSE(folder_path, corpus, label, limit=None):
  puncts = set(string.punctuation)
  counter = 0
  texts = set()

  folder_path = os.path.join(folder_path, 'BitGCSE')
  for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in [f for f in filenames if str(f).endswith('.html')]:

      path = os.path.join(str(dirpath), str(filename))

      with open(path) as f:
        try:
          soup = BeautifulSoup(f, "lxml")
          l = soup.select('#bs-content-rb-bite p')

          if l:
            all_sents = []
            for p in l:
              text = p.get_text().replace("\n", " ")

              # clear html comments, javascript and make sure you get english
              # sentences
              if '<!--' not in text:
                text = " ".join(text.split())
                if "Your web browser does not have JavaScript switched on at the moment." not in text:
                  if "In order to see this content you need to have both Javascript enabled and Flash installed." not in text:
                    if 'You will not be able to see this content until you have JavaScript switched on.' not in text:
                      if 'You have disabled Javascript, or are not running Javascript on this browser.' not in text:
                        if 'Go to the WebWise Flash install guide' not in text:
                          sents = nltk.sent_tokenize(text)
                          punctuation = False
                          for sent in sents:
                            for punct in puncts:
                              if punct in sent:
                                punctuation = True
                                break
                          if len(sents) >= 1 and punctuation:

                            # Remove html lists
                            if sents[-1].endswith(':'):
                              sents = sents[:-1]
                            if langid.classify(sents[0])[0] == 'en':
                              all_sents.extend(sents)

            # just take the articles with enough text since there are enough
            # articles in this subcorpus
            if len(all_sents) > 5:  # TODO consider changing or making param
              text = " ".join(all_sents)
              if text not in texts:
                if limit is None or counter < limit:
                  corpus.append([text, label, path])
                  texts.add(text)
                  # print("Extracting text num ", str(counter + 1), ': ', text)
                  # else:
                  #   break
                  counter += 1
        except Exception as e:
          pass
  print(f'Extracted {counter + 1} documents')
  return corpus


def get_BitKS3(folder_path, corpus, label, limit=None):
  puncts = set(string.punctuation)
  counter = 0
  texts = set()

  folder_path = os.path.join(folder_path, 'BitKS3')
  for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in [f for f in filenames if str(f).endswith(".html")]:
      path = os.path.join(str(dirpath), str(filename))
      with open(path) as f:

        try:
          soup = BeautifulSoup(f, "lxml")
          l = soup.select('.contentbox_content p')
          if l:
            all_sents = []
            for p in l:
              text = p.get_text().replace("\n", " ")
              # clear html comments, javascript and make sure you get english
              # sentences
              if '<!--' not in text:
                text = " ".join(text.split())
                if "Your web browser does not have JavaScript switched on at the moment." not in text:
                  if "In order to see this content you need to have both Javascript enabled and Flash installed." not in text:
                    if 'You will not be able to see this content until you have JavaScript switched on.' not in text:
                      sents = nltk.sent_tokenize(text)
                      punctuation = False
                      for sent in sents:
                        for punct in puncts:
                          if punct in sent:
                            punctuation = True
                            break
                      if len(sents) >= 1 and punctuation:
                        if langid.classify(sents[0])[0] == 'en':
                          all_sents.extend(sents)
            if len(all_sents) >= 1:
              text = " ".join(all_sents)
              if text not in texts:
                if limit is None or counter < limit:
                  corpus.append([text, label, path])
                  texts.add(text)
                  # print("Extracting text num ", str(counter + 1))
                  # else:
                  #   break
                  counter += 1
        except BaseException as e:
          pass
  print(f'Extracted {counter + 1} documents')
  return corpus


def get_WRLevel(folder_path, level, corpus, label, limit=None):
  puncts = set(string.punctuation)
  counter = 0
  texts = set()

  folder_path = os.path.join(folder_path, 'WRLevel')

  for dirpath, dirnames, filenames in os.walk(folder_path + str(level)):
    for filename in [f for f in filenames if str(f).endswith(".aspx")]:

      path = os.path.join(str(dirpath), str(filename))
      with open(path) as f:

        try:
          soup = BeautifulSoup(f, "lxml")
          l = soup.select('#txtArticleContent p')
          if l:
            all_sents = []
            for p in l:
              text = p.get_text().replace("\n", " ")
              # clear code comments and make sure you get english sentences
              if '<!--' not in text:
                text = " ".join(text.split())
                sents = nltk.sent_tokenize(text)
                punctuation = False
                for sent in sents:
                  for punct in puncts:
                    if punct in sent:
                      punctuation = True
                      break

                if len(sents) > 1 and punctuation:

                  if langid.classify(sents[0])[0] == 'en':
                    all_sents.extend(sents)
            if len(all_sents) >= 1:
              text = " ".join(all_sents)
              if text not in texts:
                if limit is None or counter < limit:
                  corpus.append([text, label, path])
                  texts.add(text)
                  # print("Extracting text num ", str(counter + 1), ': ', text)
                #    else:
                #        break
                  counter += 1
        except BaseException:
          pass
  print(f'Extracted {counter + 1} documents')
  return corpus


def getGroups(limit=None) -> Dict[str, List[List[Union[str, int]]]]:
  groups = {}
  # rawCorpus = os.path.join(os.pardir, 'corpora', 'rawCorpora', 'WeeBit')
  # TODO figure out why this can't go up one directory
  # limit = 600 was original
  # TODO In Martinc 2019, they just take the first 600 from each class
  # I will do that for comparison, but I should also try selecting randomly/shuffling
  print(f'limit is {limit}')
  rawCorpus = os.path.join('corpora/rawCorpora/WeeBit/')
  print('BitKS3')
  groups['BitKS3'] = get_BitKS3(rawCorpus, [], 'BitKS3', limit)
  print('BitGCSE')
  groups['BitGCSE'] = get_BitGCSE(rawCorpus, [], 'BitGCSE', limit)
  print('loading WRLevel2')
  groups['WRLevel2'] = get_WRLevel(rawCorpus, 2, [], 'WRLevel2', limit)
  print('loading WRLevel3')
  groups['WRLevel3'] = get_WRLevel(rawCorpus, 3, [], 'WRLevel3', limit)
  print('loading WRLevel4')
  groups['WRLevel4'] = get_WRLevel(rawCorpus, 4, [], 'WRLevel4', limit)

  return groups
