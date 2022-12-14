import torch
import torch.nn as nn


class FeedforwardNeuralNetModel(nn.Module):
    """
    Feed-forward Neural Network model with one hidden layer
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float
    ):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Linear function
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Linear function 2
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Linear function 3 (readout)
        self.relu = nn.ReLU()  # Non-linearity
        self.dropout = nn.Dropout(dropout_rate)  # Dropout

    def forward(self, input: torch.Tensor):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
