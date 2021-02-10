"""doc
# pytorch_mjolnir.Experiment / mjolnir_experiment

> A lightning module that runs an experiment in a managed way.

There is first the Experiment base class from wich all experiments must inherit (directly or indirectly).
"""
import os
from typing import Any, Tuple
import pytorch_lightning as pl
import torch
import argparse
from torch.utils.data.dataloader import DataLoader
from time import time
from datetime import datetime
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data.dataset import IterableDataset


def _generate_version() -> str:
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H.%M.%S')


def run(experiment_class):
    """
    You can use this main function to make your experiment runnable with command line arguments.

    Simply add this to the end of your experiment.py file:

    ```python
    if __name__ == "__main__":
        from pytorch_mjolnir import run
        run(MyExperiment)
    ```

    Then you can call your python file from the command line and use the help to figure out the parameters.
    ```bash
    python my_experiment.py --help
    ```
    """
    gpus = 1
    if "SLURM_GPUS" in os.environ:
        gpus = int(os.environ["SLURM_GPUS"])
    nodes = 1
    if "SLURM_NODES" in os.environ:
        nodes = int(os.environ["SLURM_NODES"])
    output_path = "logs"
    if "RESULTS_PATH" in os.environ:
        output_path = os.environ["RESULTS_PATH"]
    
    parser = argparse.ArgumentParser(description='The main entry point for the script.')
    parser.add_argument('--name', type=str, required=True, help='The name for the experiment.')
    parser.add_argument('--version', type=str, required=False, default=None, help='The version that should be used (defaults to timestamp).')
    parser.add_argument('--output', type=str, required=False, default=output_path, help='The name for the experiment (defaults to $RESULTS_PATH or "logs").')
    parser.add_argument('--gpus', type=int, required=False, default=gpus, help='Number of GPUs that can be used.')
    parser.add_argument('--nodes', type=int, required=False, default=nodes, help='Number of nodes that can be used.')
    parser.add_argument('--resume_checkpoint', type=str, required=False, default=None, help='A specific checkpoint to load. If not provided it tries to load latest if any exists.')
    args, other_args = parser.parse_known_args()

    experiment = experiment_class(**other_args)
    experiment.run_experiment(name=args.name, version=args.version, output_path=args.output, resume_checkpoint=args.resume_checkpoint, gpus=args.gpus, nodes=args.nodes)


class Experiment(pl.LightningModule):
    """
    An experiment base class.

    All experiments must inherit from this.
    
    ```python
    from pytorch_mjolnir import Experiment
    class MyExperiment(Experiment):
        [...]
    ```
    """
    def run_experiment(self, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), resume_checkpoint=None):
        """
        Run the experiment.

        :param name: The name of the family of experiments you are conducting.
        :param gpus: The number of gpus used for training.
        :param nodes: The number of nodes used for training.
        :param version: The name for the specific run of the experiment in the family (defaults to a timestamp).
        :param output_path: The path where to store the outputs of the experiment (defaults to the current working directory).
        :param resume_checkpoint: The path to the checkpoint that should be resumed (defaults to None).
            In case of None this searches for a checkpoint in {output_path}/{name}/{version}/checkpoints and resumes it.
            Without defining a version this means no checkpoint can be found as there will not exist a  matching folder.
        """
        if version is None:
            version = _generate_version()
        if resume_checkpoint is None:
            resume_checkpoint = self._find_checkpoint(name, version, output_path)
        trainer = pl.Trainer(
            default_root_dir=output_path,
            max_epochs=getattr(self.hparams, "max_epochs", 1000),
            gpus=gpus,
            num_nodes=nodes,
            logger=TensorBoardLogger(
                save_dir=output_path, version=version, name=name
            ),
            resume_from_checkpoint=resume_checkpoint
        )
        trainer.fit(self)

    def _find_checkpoint(self, name, version, output_path):
        resume_checkpoint = None
        checkpoint_folder = os.path.join(output_path, name, version, "checkpoints")
        if os.path.exists(checkpoint_folder):
            checkpoints = sorted(os.listdir(checkpoint_folder))
            if len(checkpoints) > 0:
                resume_checkpoint = os.path.join(checkpoint_folder, checkpoints[-1])
        return resume_checkpoint

    def load_data(self, stage=None) -> Tuple[Any, Any]:
        """
        **ABSTRACT:** Load the data for training and validation.

        :return: A tuple of the train and val dataset.
        """
        raise NotImplementedError("Must be implemented by inheriting classes.")

    def training_step(self, batch, batch_idx):
        """
        Executes a training step.

        By default this calls the step function.
        :param batch: A batch of training data received from the train loader.
        :param batch_idx: The index of the batch.
        """
        feature, target = batch
        return self.step(feature, target, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        Executes a validation step.

        By default this calls the step function.
        :param batch: A batch of val data received from the val loader.
        :param batch_idx: The index of the batch.
        """
        feature, target = batch
        return self.step(feature, target, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        """
        This function is called after all training steps.

        It accumulates the loss into a val_loss which is logged in the end.
        """
        avg_loss = torch.tensor([x for x in val_step_outputs]).mean()
        return {'val_loss': avg_loss}

    def setup(self, stage=None):
        """
        This function is for setting up the training.

        The default implementation calls the load_data function and
        stores the result in self.train_data and self.val_data.
        (It is called once per process.)
        """
        self.train_data, self.val_data = self.load_data(stage=stage)

    def train_dataloader(self):
        """
        Create a training dataloader.

        The default implementation wraps self.train_data in a Dataloader.
        """
        shuffle = True
        if isinstance(self.train_data, IterableDataset):
            shuffle = False
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=shuffle)

    def val_dataloader(self):
        """
        Create a validation dataloader.

        The default implementation wraps self.val_data in a Dataloader.
        """
        shuffle = True
        if isinstance(self.val_data, IterableDataset):
            shuffle = False
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=shuffle)
