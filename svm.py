from typing import *

import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
from features.NBTfidf import NBTfidfVectorizer
import csv
from math import sqrt
warnings.filterwarnings('ignore')


from common import DataList


def train(
        train_data: DataList,
        test_data: DataList,
        features: List[str],  # TODO make feature list object
        params=None, model='svm', add_tfidf=False):
  # TODO make this type
  # add this feature exclusion to magpie training
  # TODO add this filtering for cnn
  def extractFeatures(example: Dict[str, Any]) -> List[float]:
    pairs = filter(
        lambda pair: (not isinstance(pair[1], str)) and pair[0] != 'label',
        example.items())
    return [pair[1] for pair in pairs]

  def extractFeatureNames(example: Dict[str, Any]) -> List[str]:
    pairs = filter(
        lambda pair: (not isinstance(pair[1], str)) and pair[0] != 'label',
        example.items())
    return [pair[0] for pair in pairs]
    # pairs = filter(
  # print(f'len of 1 {train_data[0]}')
    #     lambda pair: pair[0] in features,
    #     example.items())
    # return [pair[1] for pair in pairs]
  # print(set(train_data[0].keys().))
  if add_tfidf:
    vectorizer = NBTfidfVectorizer()
    vectorizer.fit(np.array([x['text'] for x in train_data]),
                   y=np.array([x['label'] for x in train_data]))
    train_texts = np.array([x['text'] for x in train_data])
    test_texts = np.array([x['text'] for x in test_data])
    X_train = vectorizer.transform(train_texts)
    # TODO this is temp until I can fuse tfidf with older features
    X_test = vectorizer.transform(test_texts)
    # for el in train_data:
    #   el = vectorizer.transform([el['text']])
    # for el in test_data:
    #   el['nbtfid'] = vectorizer.transform([el['text']])
  else:
    feature_names = extractFeatureNames(train_data[0])
    X_train = np.vstack([extractFeatures(example)
                         for example in train_data])
    X_test = np.vstack([extractFeatures(example)
                        for example in test_data])
  # X = np.vstack([extractFeatures(example)
  #                for example in test_data + train_data])
    print(X_train.shape)
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
  le = preprocessing.LabelEncoder()
  y_train_str = [example['label'] for example in train_data]
  y_test_str = [example['label'] for example in test_data]
  le.fit(y_train_str + y_test_str)
  y_train = np.vstack(le.transform(y_train_str))
  y_test = np.vstack(le.transform(y_test_str))
  # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1],
  #                      'C': [1]},
  #                     {'kernel': ['linear'], 'C': [1]}]  # TODO make this param
  # clf = GridSearchCV(SVC(cache_size=1000), tuned_parameters, cv=5,  # TODO make this params
  #                    scoring='r2', n_jobs=-1)
  if model == 'LinearSVC':
    clf = LinearSVC()
  elif model == 'LogisticRegression':
    clf = LogisticRegression(n_jobs=-1, solver='sag')
  else:
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]  # TODO make this param
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,  # TODO make this param
                       scoring='f1_weighted', n_jobs=-1)  # TODO consider wheter we want to optimize for
  clf.fit(X_train, y_train)
  y_true, y_pred = y_test, clf.predict(X_test)
  report = sklearn.metrics.classification_report(
      y_true, y_pred)
  weighted_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
  macro_f1 = sklearn.metrics.f1_score(
      y_true, y_pred, average='macro')
  micro_f1 = sklearn.metrics.f1_score(
      y_true, y_pred, average='micro')
  rsme = sqrt(sklearn.metrics.mean_squared_error(
      y_true, y_pred))
  # return clf.best_estimator_, report, weighted_f1, macro_f1 # TODO go
  # between these two for svm and linear
  if model == 'LinearSVC' or model == 'LogisticRegression':
    avg_weights = np.mean(clf.coef_, axis=0)
    with open('weights.csv', 'w', newline='') as csv_file:
      weight_csv = csv.writer(csv_file)
      weight_csv.writerow(['feature', 'weight'])
      weights_zipped = zip(feature_names, avg_weights)
      for name, weight in weights_zipped:
        weight_csv.writerow([name, weight])
  return (clf if model !=
          "svm" else clf.best_estimator_), report, weighted_f1, macro_f1, micro_f1, rsme
