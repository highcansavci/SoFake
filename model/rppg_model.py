import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.rnn1 = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )
        self.rnn3 = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )
        self.rnn4 = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )
        self.rnn5 = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )

        self.fc1 = nn.Linear(self.M, self.K)
        self.fc2 = nn.Linear(self.M, self.K)
        self.fc3 = nn.Linear(self.M, self.K)
        self.fc4 = nn.Linear(self.M, self.K)
        self.fc5 = nn.Linear(self.M, self.K)

        self.classifier = nn.Linear(self.K * 5, 1)

    def forward(self, X):
        h0 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        c0 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        h1 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        c1 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        h2 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        c2 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        h3 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        c3 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        h4 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)
        c4 = torch.randn(X.size(0), self.M, device=self.device, dtype=torch.float)

        x1, x2, x3, x4, x5 = torch.tensor_split(X, 5, dim=1)
        out1, _ = self.rnn1(x1, (h0, c0))
        out2, _ = self.rnn2(x2, (h1, c1))
        out3, _ = self.rnn3(x3, (h2, c2))
        out4, _ = self.rnn4(x4, (h3, c3))
        out5, _ = self.rnn5(x5, (h4, c4))

        # we only want h(T) at the final time step
        # N x M -> N x K
        out1 = self.fc1(out1[-1, :])
        out2 = self.fc2(out2[-1, :])
        out3 = self.fc3(out3[-1, :])
        out4 = self.fc4(out4[-1, :])
        out5 = self.fc5(out5[-1, :])

        out = torch.cat((out1, out2, out3, out4, out5))
        out = self.classifier(out)
        return out