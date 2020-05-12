import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from han.src.utils import get_max_lengths, get_evaluation
from han.src.dataset import MyDataset
from han.src.hierarchical_att_model import HierAttNet
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from typing import Any


def get_args():
  parser = argparse.ArgumentParser(
      """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
  parser.add_argument("--batch_size", type=int, default=64)

  parser.add_argument("--num_epoches", type=int, default=20)
  parser.add_argument("--lr", type=float, default=0.0001)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--word_hidden_size", type=int, default=200)
  parser.add_argument("--sent_hidden_size", type=int, default=100)
  parser.add_argument(
      "--es_min_delta",
      type=float,
      default=0.01,
      help="Early stopping's parameter: minimum change loss to qualify as an improvement")
  parser.add_argument(
      "--es_patience",
      type=int,
      default=5,
      help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
  parser.add_argument("--train_set", type=str, default="./data/newsela.csv")
  parser.add_argument("--test_set", type=str, default="./data/newsela.csv")
  parser.add_argument(
      "--test_interval",
      type=int,
      default=1,
      help="Number of epoches between testing phases")
  parser.add_argument(
      "--vocab_path",
      type=str,
      default="../vocab/newsela_vocab.pk")
  parser.add_argument("--saved_path", type=str, default="trained_models")
  parser.add_argument("--output", type=str, default="newsela_new_test")
  parser.add_argument('configs/Newsela/han.yaml', default='test')

  args = parser.parse_args()
  return args


def train(
        opt,
        task,
        data_path=None,
        fed_data=None,
        k_fold=0,
        subsample=1,
        overwrite=True,
        config=''):
  if torch.cuda.is_available():
    torch.cuda.manual_seed(2019)
  else:
    torch.manual_seed(2019)
  output_file = open(opt.saved_path + "logs.txt", "w")
  output_file.write("Model's parameters: {}".format(vars(opt)))
  training_params = {"batch_size": opt.batch_size,
                     "shuffle": False,
                     "drop_last": True}
  test_params = {"batch_size": opt.batch_size,
                 "shuffle": False,
                 "drop_last": False}

  max_word_length, max_sent_length = get_max_lengths(
      opt.train_set, fed_data=fed_data)
  print("Max words: ", max_word_length, "Max sents: ", max_sent_length)

  if fed_data is None:
    df_data = pd.read_csv(data_path, encoding='utf8', sep='\t')
    # df_data = df_data.sample(frac=1, random_state=2019)
    # print(df_data.shape)
  else:
    print('han fed data')
    df_train, df_test = pd.DataFrame(
        fed_data[0], columns=[
            'text', 'readability']), pd.DataFrame(
        fed_data[1], columns=[
            'text', 'readability'])
    print(df_train.shape, df_test.shape)

  kf = model_selection.KFold(n_splits=5)

  predicted_all_folds = []
  true_all_folds = []
  counter = 0
  accuracies_all_folds = []
  precision_all_folds = []
  recall_all_folds = []
  f1_all_folds = []

  if fed_data is None:
    kf = model_selection.KFold(n_splits=5)
    iterator = kf.split(df_data)
  else:
    iterator = [(0, 0)]

  for train_index, test_index in iterator:

    counter += 1

    print()
    print("CV fold: ", counter)
    print()

    if os.path.exists(opt.vocab_path):
      os.remove(opt.vocab_path)

    if fed_data is None:
      df_train, df_test = df_data.iloc[train_index], df_data.iloc[test_index]
      sep_idx = int(df_test.shape[0] / 2)
      df_valid = df_test[:sep_idx]
      df_test = df_test[sep_idx:]
      print(
          "Train size: ",
          df_train.shape,
          "Valid size: ",
          df_valid.shape,
          "Test size: ",
          df_test.shape)

    training_set = MyDataset(
        df_train,
        opt.vocab_path,
        task,
        max_sent_length,
        max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(
        df_test,
        opt.vocab_path,
        task,
        max_sent_length,
        max_word_length)
    test_generator = DataLoader(test_set, **test_params)

    # valid_set = MyDataset(
    #     df_valid,
    #     opt.vocab_path,
    #     task,
    #     max_sent_length,
    #     max_word_length)
    # valid_generator = DataLoader(valid_set, **test_params)

    model = HierAttNet(
        opt.word_hidden_size,
        opt.sent_hidden_size,
        opt.batch_size,
        training_set.num_classes,
        opt.vocab_path,
        max_sent_length,
        max_word_length)

    if torch.cuda.is_available():
      model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    best_loss = 1e5
    best_epoch = 0
    num_iter_per_epoch = len(training_generator)

    for epoch in range(opt.num_epoches):
      model.train()
      for iter, (feature, label) in enumerate(training_generator):
        if torch.cuda.is_available():
          feature = feature.cuda()
          label = label.cuda()
        optimizer.zero_grad()
        model._init_hidden_state()
        predictions = model(feature)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        training_metrics = get_evaluation(
            label.cpu().numpy(),
            predictions.cpu().detach().numpy(),
            list_metrics=["accuracy"])
        print(
            "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                training_metrics["accuracy"]))

      if epoch % opt.test_interval == 0:
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for te_feature, te_label in valid_generator:
          num_sample = len(te_label)
          if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
          with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
          te_loss = criterion(te_predictions, te_label)
          loss_ls.append(te_loss * num_sample)
          te_label_ls.extend(te_label.clone().cpu())
          te_pred_ls.append(te_predictions.clone().cpu())
        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(
            te_label, te_pred.numpy(), list_metrics=[
                "accuracy", "confusion_matrix"])

        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            opt.num_epoches,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        if te_loss + opt.es_min_delta < best_loss:
          best_loss = te_loss
          best_epoch = epoch
          print('Saving model')
          torch.save(
              model,
              opt.saved_path +
              os.sep +
              f"{config.corpus}_{config.name.replace('/', '_')}_k_{k_fold}_subsample_{subsample}_han.bin")

        # Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
          print(
              "Stop training at epoch {}. The lowest loss achieved is {}".format(
                  epoch, best_loss))
          break

    print()
    print('Evaluation: ')
    print()

    model.eval()
    model = torch.load(opt.saved_path + os.sep + "whole_model_han")
    loss_ls = []
    te_label_ls = []
    te_pred_ls = []
    for te_feature, te_label in test_generator:
      num_sample = len(te_label)
      if torch.cuda.is_available():
        te_feature = te_feature.cuda()
        te_label = te_label.cuda()
      with torch.no_grad():
        model._init_hidden_state(num_sample)
        te_predictions = model(te_feature)
      te_loss = criterion(te_predictions, te_label)
      loss_ls.append(te_loss * num_sample)
      te_label_ls.extend(te_label.clone().cpu())
      te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0)
    te_label = np.array(te_label_ls)
    test_metrics = get_evaluation(
        te_label,
        te_pred.numpy(),
        list_metrics=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "confusion_matrix"])

    true = te_label
    preds = np.argmax(te_pred.numpy(), -1)
    predicted_all_folds.extend(preds)
    true_all_folds.extend(true)

    f1 = f1_score(true, preds, average='weighted')
    macro_f1 = f1_score(true, preds, average='macro')
    micro_f1 = f1_score(true, preds, average='micro')
    rmse = sqrt(mean_squared_error(true, preds))

    print("Test set accuracy: {}".format(test_metrics["accuracy"]))
    print("Test set precision: {}".format(test_metrics["precision"]))
    print("Test set recall: {}".format(test_metrics["recall"]))
    print("Test set f1: {}".format(test_metrics["f1"]))
    print("Test set cm: {}".format(test_metrics["confusion_matrix"]))

    accuracies_all_folds.append(test_metrics["accuracy"])
    precision_all_folds.append(test_metrics["precision"])
    recall_all_folds.append(test_metrics["recall"])
    f1_all_folds.append(test_metrics["f1"])
    print()

  print()
  print("Task: ", task)
  print("Accuracy: ", accuracy_score(true_all_folds, predicted_all_folds))
  print(
      'Confusion matrix: ',
      confusion_matrix(
          true_all_folds,
          predicted_all_folds))
  print('All folds accuracy: ', accuracies_all_folds)
  print('All folds precision: ', precision_all_folds)
  print('All folds recall: ', recall_all_folds)
  print('All folds f1: ', f1_all_folds)

  if fed_data is not None:
    class FakeMagpie:
      def predict_from_text(self, text: str, return_float=False):
        df_single = pd.DataFrame(
            [[text, -1]], columns=[
                'text', 'readability'])
        el_set = MyDataset(
            df_single,
            opt.vocab_path,
            task,
            max_sent_length,
            max_word_length)
        el_generator = DataLoader(el_set, **test_params)
        for te_feature, te_label in el_generator:
          num_sample = len(te_label)
          if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
          with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
        if return_float:
          return float(te_predictions.clone().cpu())
        else:
          return int(te_predictions.clone().cpu())
    return FakeMagpie(), f1, macro_f1, micro_f1, rmse


if __name__ == "__main__":
  opt = get_args()
  train(opt, 'newsela', opt.train_set)


def main(
        task,
        fed_data=None,
        k_fold=0,
        subsample=1,
        overwrite=True,
        config='') -> Any:
  opt = get_args()
  return train(opt,
               task,
               data_path=None,
               fed_data=fed_data,
               k_fold=k_fold,
               subsample=subsample,
               overwrite=overwrite,
               config=config)
