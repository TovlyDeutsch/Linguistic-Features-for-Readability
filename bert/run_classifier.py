# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import pandas as pd
from sklearn import model_selection
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score, mean_squared_error
from math import sqrt
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



class WeebitProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text_a = row['text']
            text_b = None
            label = str(int(row['readability'])) #TODO this used to ahave a minus 2 here
            #print(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class NewselaProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        # return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text_a = row['text']
            text_b = None
            label = str(int(row['readability']))
            if label == '0':
                print(f"found 0 in create examples with row {row['readability']}")
            # if int(label) > 10 or int(label) < 0:
            #     print(label, text_a)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class OneStopEnglishProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text_a = row['text']
            text_b = None
            label = str(int(row['readability']))
            #print(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class UcbenikiProcessor(DataProcessor):

    def get_train_examples(self, data):
        """See base class."""
        return self._create_examples(data, "train")

    def get_dev_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, row in lines.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text_a = row['text']
            text_b = None
            label = str(int(row['readability']))
            #print(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples




def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if example.label == '-2':
            print(example)
        if example.label not in label_map:
            print(label_map)
            print(example.label)
            print(label_list)
        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main(fed_data=None, k_fold=0, subsample=1, overwrite=True, config='', task_name='weebit') -> Any:
    parser = argparse.ArgumentParser()
    print(f'task name is {task_name}')


    ## Required parameters
    parser.add_argument("--data_dir",
                        default='',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=task_name,
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='./results/' + task_name,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('-a', default=0)
    parser.add_argument('configs/WeeBit/transformer600.yaml', default='test')

    args = parser.parse_args()

    output_model_file = os.path.join(args.output_dir, f"{config.corpus}_{config.name.replace('/', '_')}_k_{k_fold}_subsample_{subsample}_pytorch_model.bin")
    if overwrite == False and os.path.exists(output_model_file):
        args.do_train = False

    processors = {
        "weebit": WeebitProcessor,
        "newsela": NewselaProcessor,
        "onestopenglish": OneStopEnglishProcessor,
        "ucbeniki": UcbenikiProcessor,
    }

    num_labels_task = {
        "weebit": 5,
        "newsela": 11,
        "onestopenglish": 3,
        "ucbeniki": 13,
    }

    input_paths = {
        "weebit": os.path.join("data/WeeBit/weebit_reextracted.tsv"),
        "newsela": os.path.join("../../data/newsela.csv"),
        "onestopenglish": os.path.join("../../data/oneStopEnglish/onestopenglish_docs.csv"),
        "ucbeniki": os.path.join("../../data/Ucbeniki_oznaceni/ucbeniki_classification_docs_cleaned.csv"),
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    input_path = input_paths[task_name]
    if fed_data is None:
        df_data = pd.read_csv(input_path, encoding='utf8', sep='\t')
        # df_data = df_data.sample(frac=1, random_state=2019)
        print(df_data.shape)
    else:
        print('transformer fed data')
        df_train, df_test = pd.DataFrame(fed_data[0], columns=['text', 'readability']), pd.DataFrame(fed_data[1], columns=['text', 'readability'])
        print(df_train.shape, df_test.shape)

    predicted_all_folds = []
    true_all_folds = []

    accuracies_all_folds = []
    precision_all_folds = []
    recall_all_folds = []
    f1_all_folds = []

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if fed_data is None:
        kf = model_selection.KFold(n_splits=5)
        iterator = kf.split(df_data)
    else:
        iterator = [(0, 0)]

    for train_index, test_index in iterator:
        if fed_data is None:
            df_train, df_test = df_data.iloc[train_index], df_data.iloc[test_index]
            sep_idx = int(df_test.shape[0] / 2)
            df_test = df_test[sep_idx:]

        if args.do_train:
            train_examples = processor.get_train_examples(df_train)
            num_train_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Prepare model
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                  num_labels = num_labels)
        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16 and args.do_train:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        elif args.do_train:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        if args.do_train:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            model.train()
            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)

        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
        model.to(device)

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            eval_examples = processor.get_dev_examples(df_test)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            predicted_all = []
            true_all = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)

                preds = np.argmax(logits, axis=1).tolist()
                true = label_ids.tolist()
                predicted_all.extend(preds)
                true_all.extend(true)
                predicted_all_folds.extend(preds)
                true_all_folds.extend(true)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            print("Task: ", task_name)
            accuracy_fold = accuracy_score(true_all, predicted_all)
            precision = precision_score(true_all, predicted_all, average='weighted')
            recall = recall_score(true_all, predicted_all, average='weighted')
            f1 =  f1_score(true_all, predicted_all, average='weighted')
            macro_f1 =  f1_score(true_all, predicted_all, average='macro')
            micro_f1 =  f1_score(true_all, predicted_all, average='micro')
            rmse =  sqrt(mean_squared_error(true_all, predicted_all))
            accuracies_all_folds.append(accuracy_fold)
            precision_all_folds.append(precision)
            recall_all_folds.append(recall)
            f1_all_folds.append(f1)

            print("Accuracy fold: ", accuracy_score(true_all, predicted_all))
            print("Precision fold: ", precision_score(true_all, predicted_all, average='weighted'))
            print("Recall fold: ", recall_score(true_all, predicted_all, average='weighted'))
            print("F1 fold: ",f1_score(true_all, predicted_all, average='weighted'))
            print('Confusion matrix: ', confusion_matrix(true_all, predicted_all))
            if fed_data is not None:
                # TODO return actual clf instead of None for use with other features
                class FakeMagpie:
                    def predict_from_text(self, text: str, return_float=False):
                        inference_examples = [InputExample(guid="test-0", text_a=text, text_b=None, label='0')]
                        inference_features = convert_examples_to_features(inference_examples, label_list, args.max_seq_length, tokenizer)
                        all_inference_ids = torch.tensor([f.input_ids for f in inference_features], dtype=torch.long)
                        all_inference_mask = torch.tensor([f.input_mask for f in inference_features], dtype=torch.long)
                        all_segment_ids = torch.tensor([f.segment_ids for f in inference_features], dtype=torch.long)
                        all_label_ids = torch.tensor([f.label_id for f in inference_features], dtype=torch.long)
                        inference_data = TensorDataset(all_inference_ids, all_inference_mask, all_segment_ids, all_label_ids)
                
                        inference_dataloader = DataLoader(inference_data)
                        for batch in inference_dataloader:
                            input_ids, input_mask, segment_ids, label_ids = batch
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        logits = model(input_ids, segment_ids, input_mask)
                        logits = logits.detach().cpu().numpy()
                        preds = np.argmax(logits, axis=1).tolist()
                        if return_float:
                            return float(preds[0])
                        else:
                            return int(preds[0])
                return FakeMagpie(), f1, macro_f1, micro_f1, rmse

        del loss
        del tr_loss
        del tmp_eval_loss
        del eval_loss
        del eval_data
        del eval_dataloader
        del optimizer
        del logits
        torch.cuda.empty_cache()

    print("Task: ", task_name)
    # TODO be aware this is not the same as weighted accuracy, I think
    print("Accuracy: ", accuracy_score(true_all_folds, predicted_all_folds))
    print("Precision: ", precision_score(true_all_folds, predicted_all_folds, average='weighted'))
    print("Recall: ", recall_score(true_all_folds, predicted_all_folds, average='weighted'))
    print("F1: ", f1_score(true_all_folds, predicted_all_folds, average='weighted'))
    print('Confusion matrix: ', confusion_matrix(true_all_folds, predicted_all_folds))

    print('All folds accuracy: ', accuracies_all_folds)
    print('All folds precision: ', precision_all_folds)
    print('All folds recall: ', recall_all_folds)
    print('All folds f1: ', f1_all_folds)

if __name__ == "__main__":
    main()




