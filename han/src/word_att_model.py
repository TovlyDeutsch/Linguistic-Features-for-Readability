import torch
import torch.nn as nn
import torch.nn.functional as F
from han.src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv
import pickle


class WordAttNet(nn.Module):
  def __init__(self, vocab_path, hidden_size=50):
    super(WordAttNet, self).__init__()
    embed_size = 200
    dict_len = len(pickle.load(open(vocab_path, 'rb'))) + 1

    self.word_weight = nn.Parameter(
        torch.Tensor(2 * hidden_size, 2 * hidden_size))
    self.word_bias = nn.Parameter(torch.zeros(1, 2 * hidden_size))
    self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

    self.lookup = nn.Embedding(
        num_embeddings=dict_len,
        embedding_dim=embed_size)
    self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True, dropout=0.2)
    self._create_weights(mean=0.0, std=0.05)

  def _create_weights(self, mean=0.0, std=0.05):
    self.word_weight.data.normal_(mean, std)
    self.context_weight.data.normal_(mean, std)

  def forward(self, input, hidden_state):

    output = self.lookup(input)
    # feature output and hidden state output
    f_output, h_output = self.gru(output.float(), hidden_state)
    output = matrix_mul(f_output, self.word_weight, self.word_bias)
    output = matrix_mul(output, self.context_weight).permute(1, 0)
    output = F.softmax(output, dim=-1)
    output = element_wise_mul(f_output, output.permute(1, 0))

    return output, h_output
