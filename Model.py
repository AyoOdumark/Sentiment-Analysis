import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, state_vec_size):
        super(Encoder, self).__init__()
        self.hidden_size = state_vec_size
        self.i2h = nn.Linear(input_size + state_vec_size, state_vec_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        state_vec = F.relu(self.i2h(combined))
        return state_vec

    def init_hidden(self):
        return torch.zeros((1, self.hidden_size))


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, input_vec):
        output = F.relu(self.hidden(input_vec))
        output = F.sigmoid(self.output(output))
        return output


