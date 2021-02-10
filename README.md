# Pytorch Mjolnir

> Thinking in experiments made simpler for pytorch-lightning.
> So that no experiment is wasted and you can iterate faster.

On pypi some links are broken -> [Visit the project on github](https://github.com/penguinmenac3/pytorch-mjolnir).

## Getting Started

Simply pip install

```bash
pip install pytorch_mjolnir
```

Then read the [Documentation](docs/README.md) of its API containing also examples.

## Writing an Experiment

When you already have your code split into a model and a loss, it is easy to convert it to a mjolnir experiment.
Simply use the SupervisedExperiment experiment or any of the precanned experiments.
(Learn more in the [documentation](docs/README.md))

```python
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split

from pytorch_mjolnir import SupervisedExperiment


class MNISTExperiment(SupervisedExperiment):
    def __init__(self, learning_rate=1e-3, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.model = MyModel()  # Any old pytorch model -> preds = self.model(*features)
        self.loss = MyLoss()    # Any loss -> loss = self.loss(preds, targets)

    def prepare_data(self):
        # Prepare the data once (no state allowed due to multi-gpu/node setup.)
        MNIST(".datasets", train=True, download=True)

    def load_data(self, stage=None) -> Tuple[Any, Any]:
        # Load your datasets.
        dataset = MNIST(".datasets", train=True, download=False, transform=transforms.ToTensor())
        return random_split(dataset, [55000, 5000])

    def configure_optimizers(self):
        # Create an optimizer to your liking.
        return Adam(self.parameters(), lr=self.hparams.learning_rate)


# Run the experiment when the script is executed.
if __name__ == "__main__":
    from pytorch_mjolnir import run
    run(MNISTExperiment)
```

## Running an experiment

**Local**: Simply run your experiment py file from the command line.
It has lots of parameters to customize its behaviour.
```bash
python examples/autoencoder.py --name=Autoencoder
```

**Remote/SLURM**: In a cluster setting check out how the remote run command works, it might make you much more productive. You simply specify a `run.template.slurm` and a `run.template.sh` (see [examples](examples)).
```bash
mjolnir_remote examples/autoencoder.py --name=Autoencoder --host=slurm.yourcompany.com
```

## Contributing

Currently there are no guidelines on how to contribute, so the best thing you can do is open up an issue and get in contact that way.
In the issue we can discuss how you can implement your new feature or how to fix that nasty bug.

To contribute, please fork the repositroy on github, then clone your fork. Make your changes and submit a merge request.

## Origin of the Name

Mjolnir is thors weapon and a catalyst for lightning. As this library is about being a catalyst for experiments with lightning, the choice.

## License

This repository is under MIT License. Please see the [full license here](LICENSE).
