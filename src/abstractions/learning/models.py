"""
models.py

Defines neural network architectures used in the abstraction discovery pipeline.
"""

from torch import nn


class MLP(nn.Module):
    """
    A simple feedforward multi-layer perceptron with one hidden layer.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input features.
    output_dim : int
        The dimensionality of the output.

    Attributes
    ----------
    linear_1 : nn.Linear
        First linear layer mapping input_dim to input_dim.
    activation : nn.ReLU
        ReLU activation function.
    linear_2 : nn.Linear
        Second linear layer mapping input_dim to output_dim.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class Autoencoder(nn.Module):
    """
    A deep autoencoder neural network for learning compressed latent
    representations of input feature vectors.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input features.
    hidden_dim : int
        The dimensionality of the latent encoding.

    Attributes
    ----------
    activation : nn.ReLU
        ReLU activation function used between layers.
    encoder : nn.Sequential
        The encoder network mapping input_dim to hidden_dim.
    decoder : nn.Sequential
        The decoder network reconstructing inputs from hidden_dim.
    """

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
        """
        Forward pass of the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            encoding: Latent representation of shape (batch_size, hidden_dim).
            reconstruction: Reconstructed input of shape (batch_size, input_dim).
        """
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return encoding, reconstruction
