import os
from bs4 import BeautifulSoup
import nltk
import langid
import pandas as pd
import string
import argparse



def get_BitGCSE(folder_path, corpus):
    puncts = set(string.punctuation)
    counter = 0
    texts = set()

    folder_path = os.path.join(folder_path, 'BitGCSE')

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in [f for f in filenames if f.endswith(".html")]:

            path = os.path.join(dirpath, filename)

            with open(path) as f:
               try:
                   soup = BeautifulSoup(f, "lxml")
                   l = soup.select('#bs-content-rb-bite p')

                   if l:
                       all_sents = []
                       for p in l:
                           text = p.get_text().replace("\n", " ")

                           # clear html comments, javascript and make sure you get english sentences
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

                                                       #Remove html lists
                                                       if sents[-1].endswith(':'):
                                                           sents = sents[:-1]
                                                       if langid.classify(sents[0])[0] == 'en':
                                                           all_sents.extend(sents)

                       #just take the articles with enough text since there are enough articles in this subcorpus
                       if len(all_sents) > 5:
                           text = " ".join(all_sents)
                           if text not in texts:
                               if counter < 600:
                                   corpus.append([text, 6])
                                   texts.add(text)
                                #    print("Extracting text num ", str(counter + 1), ': ', text)
                               else:
                                   break
                               counter += 1
               except:
                   pass
    return corpus



def get_BitKS3(folder_path, corpus):
    puncts = set(string.punctuation)
    counter = 0
    texts = set()

    folder_path = os.path.join(folder_path, 'BitKS3')

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in [f for f in filenames if f.endswith(".html")]:

            path = os.path.join(dirpath, filename)
            with open(path) as f:

               try:
                   soup = BeautifulSoup(f, "lxml")
                   l = soup.select('.contentbox_content p')
                   if l:
                       all_sents = []
                       for p in l:
                           text = p.get_text().replace("\n", " ")
                           # clear html comments, javascript and make sure you get english sentences
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
                               if counter < 600:
                                   corpus.append([text, 5])
                                   texts.add(text)
                                #    print("Extracting text num ", str(counter + 1), ': ', text)
                               else:
                                   break
                               counter += 1
               except:pass
    return corpus

def get_WRLevel(folder_path, level, corpus):
   puncts = set(string.punctuation)
   counter = 0
   texts = set()

   folder_path = os.path.join(folder_path, 'WRLevel')

   for dirpath, dirnames, filenames in os.walk(folder_path + str(level)):
       for filename in [f for f in filenames if f.endswith(".aspx")]:

           path = os.path.join(dirpath, filename)
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
                               if counter < 600:
                                   corpus.append([text, level])
                                   texts.add(text)
                                #    print("Extracting text num ", str(counter + 1), ': ', text)
                               else:
                                   break
                               counter += 1
               except:
                   pass

   return corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for extracting the WeeBit corpus")
    parser.add_argument("--input_path", type=str, default="../corpora/rawCorpora/WeeBit/")
    parser.add_argument("--output_path", type=str, default="data/WeeBit/")
    args = parser.parse_args()
    corpus = []

    corpus = get_BitKS3(args.input_path, corpus)
    corpus = get_BitGCSE(args.input_path, corpus)
    corpus = get_WRLevel(args.input_path, 2, corpus)
    corpus = get_WRLevel(args.input_path, 3, corpus)
    corpus = get_WRLevel(args.input_path, 4, corpus)

    df = pd.DataFrame(corpus, columns=['text', 'readability'])
    df.to_csv(os.path.join(args.output_path, "weebit_reextracted.tsv"), sep='\t', encoding="utf8", index=False)