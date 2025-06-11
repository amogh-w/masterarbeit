from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = nn.ReLU()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            self.activation,
            nn.Linear(32, 32),
            self.activation,
            nn.Linear(32, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            self.activation,
            nn.Linear(32, 32),
            self.activation,
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return encoding, reconstruction
