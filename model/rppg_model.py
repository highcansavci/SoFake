import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, device):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size, bias=False, device=device)

    def forward(self, rnn_outputs, final_hidden_state):
        batch_size, seq_len, _ = rnn_outputs.shape
        attention_weights = self.fc1(rnn_outputs)
        attention_weights = torch.bmm(attention_weights, final_hidden_state.unsqueeze(2))
        attention_weights = F.softmax(attention_weights.squeeze(2), dim=1)
        context_vector = torch.bmm(rnn_outputs.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        pre_activation = torch.cat((context_vector, final_hidden_state), dim=1)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights


class AttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(AttnLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            device=device
        )
        self.attn = Attention(hidden_size=hidden_size, device=device)
        self.fc = nn.Linear(hidden_size, output_size, device=device)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size, device=self.device, dtype=torch.float)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size, device=self.device, dtype=torch.float)
        x, hidden_state = self.lstm(x, (h0, c0))
        final_state = hidden_state[0].view(self.num_layers, 1, x.size(0), self.hidden_size)[-1].squeeze(0)
        x, weights = self.attn(x, final_state)
        x = self.fc(x)
        return x, weights


class RPPGModel(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_layers, device):
        super(RPPGModel, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_layers  # number of rnn layers
        self.device = device

        # note: batch_first=True
        # (num_samples, sequence_length, num_features)
        # rather than:
        # (sequence_length, num_samples, num_features)
        self.rnn = AttnLSTM(
            input_size=self.D,
            hidden_size=self.M,
            output_size=self.K,
            num_layers=self.L,
            device=device
        )

        self.classifier = nn.Linear(self.K, 1, device=device)

    def forward(self, x):
        out, _ = self.rnn(x)
        # final dense layer
        out = self.classifier(out)
        return out
