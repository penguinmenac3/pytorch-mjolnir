"""doc
# Example: Autoencoder

> A sample on how to write a fully custom experiment.

See the source code [examples/autoencoder.py](../../examples/autoencoder.py)
"""
from typing import Any, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torchvision import transforms
from pytorch_mjolnir import Experiment


class AutoEncoderExperiment(Experiment):
    def __init__(
        self,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=10,
        num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros((1, 28 * 28), dtype=torch.float32)
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        """
        In lightning, forward defines the prediction/inference actions.
        """
        embedding = self.encoder(x)
        return embedding

    def step(self, feature, target, batch_idx):
        """
        Step defined the train/val loop. It is independent of forward.
        """
        feature = feature.view(feature.size(0), -1)
        z = self.encoder(feature)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, feature)
        self.log('loss/total', loss)
        self.log_resources()
        self.log_fps()
        return loss

    def prepare_data(self):
        """
        Prepare the data once (no state allowed due to multi-gpu/node setup.)
        """
        MNIST(".datasets", train=True, download=True)

    def load_data(self, stage=None) -> Tuple[Any, Any]:
        """
        Load your datasets.

        :return: Tuple of train, val dataset.
        """
        dataset = MNIST(".datasets", train=True, download=False, transform=transforms.ToTensor())
        return random_split(dataset, [55000, 5000])

    def configure_optimizers(self):
        """
        Create an optimizer to your liking.

        :return: The torch optimizer used for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main():
    """
    Manually run the training.
    """
    autoencoder = AutoEncoderExperiment()
    autoencoder.run_experiment(output_path="logs", name="autoencoder", gpus=1, nodes=1)

if __name__ == "__main__":
    main()
