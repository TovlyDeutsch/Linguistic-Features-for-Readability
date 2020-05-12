
import torch
import torch.nn as nn
import torch.nn.functional as F
from han.src.utils import matrix_mul, element_wise_mul


class LayerNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-12):

    super(LayerNorm, self).__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.bias = nn.Parameter(torch.zeros(hidden_size))
    self.variance_epsilon = eps

  def forward(self, x):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.variance_epsilon)
    return self.weight * x + self.bias


class SentAttNet(nn.Module):
  def __init__(
          self,
          sent_hidden_size,
          word_hidden_size,
          num_classes,
          batch_size):
    super(SentAttNet, self).__init__()
    print(batch_size)

    self.sent_weight = nn.Parameter(
        torch.Tensor(
            2 * sent_hidden_size,
            2 * sent_hidden_size))
    self.sent_bias = nn.Parameter(torch.zeros(1, 2 * sent_hidden_size))
    self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

    self.gru = nn.GRU(
        2 * word_hidden_size,
        sent_hidden_size,
        bidirectional=True,
        dropout=0.2)
    self.fc = nn.Linear(2 * sent_hidden_size, num_classes)

    self._create_weights(mean=0.0, std=0.05)
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()

  def _create_weights(self, mean=0.0, std=0.05):
    self.sent_weight.data.normal_(mean, std)
    self.context_weight.data.normal_(mean, std)

  def forward(self, input, hidden_state):

    f_output, h_output = self.gru(input, hidden_state)
    output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
    output_2 = matrix_mul(output, self.context_weight).permute(1, 0)
    if torch.isnan(output_2[0][0]):
      print('nan detected')
      return None

    output = F.softmax(output_2, dim=-1)
    output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
    output = self.dropout(self.fc(output))

    return output, h_output
