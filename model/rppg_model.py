import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, hidden_states):
        score_ = self.fc1(hidden_states)
        h_t = hidden_states[-1, :]
        score = torch.matmul(score_, h_t)
        attention_weights = F.softmax(score, dim=0)
        context_vector = torch.matmul(hidden_states.permute(1, 0), attention_weights)
        pre_activation = torch.cat((context_vector, h_t))
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights


class AttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(AttnLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.attn = Attention(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.randn(x.size(0), self.hidden_size, device=self.device, dtype=torch.float)
        c0 = torch.randn(x.size(0), self.hidden_size, device=self.device, dtype=torch.float)
        x, _ = self.lstm(x, (h0, c0))
        x, weights = self.attn(x)
        x = self.fc(x)
        return x, weights


class RPPGModel(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, device):
        super(RPPGModel, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = 1  # number of rnn layers
        self.device = device

        # note: batch_first=True
        # (num_samples, sequence_length, num_features)
        # rather than:
        # (sequence_length, num_samples, num_features)
        self.rnn1 = AttnLSTM(
            input_size=self.D,
            hidden_size=self.M,
            output_size=self.K,
            num_layers=self.L,
            device=device
        )

        self.rnn2 = AttnLSTM(
            input_size=self.D,
            hidden_size=self.M,
            output_size=self.K,
            num_layers=self.L,
            device=device
        )

        self.rnn3 = AttnLSTM(
            input_size=self.D,
            hidden_size=self.M,
            output_size=self.K,
            num_layers=self.L,
            device=device
        )

        self.rnn4 = AttnLSTM(
            input_size=self.D,
            hidden_size=self.M,
            output_size=self.K,
            num_layers=self.L,
            device=device
        )

        self.rnn5 = AttnLSTM(
            input_size=self.D,
            hidden_size=self.M,
            output_size=self.K,
            num_layers=self.L,
            device=device
        )

        self.fc1 = nn.Linear(self.K, self.K)
        self.fc2 = nn.Linear(self.K, self.K)
        self.fc3 = nn.Linear(self.K, self.K)
        self.fc4 = nn.Linear(self.K, self.K)
        self.fc5 = nn.Linear(self.K, self.K)

        self.classifier = nn.Linear(self.K * 5, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = torch.tensor_split(x, 5, dim=1)
        out1, _ = self.rnn1(x1)
        out2, _ = self.rnn2(x2)
        out3, _ = self.rnn3(x3)
        out4, _ = self.rnn4(x4)
        out5, _ = self.rnn5(x5)

        # we only want h(T) at the final time step
        # N x M -> N x K
        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        out3 = self.fc3(out3)
        out4 = self.fc4(out4)
        out5 = self.fc5(out5)

        out = torch.cat((out1, out2, out3, out4, out5))
        out = self.classifier(out)
        return out
