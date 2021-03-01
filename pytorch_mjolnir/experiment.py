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
import psutil
import GPUtil
from torch.utils.data.dataloader import DataLoader
from time import time
from datetime import datetime
from torch.utils.data.dataset import IterableDataset

from pytorch_mjolnir.tensorboard import TensorBoardLogger


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

    kwargs = parse_other_args(other_args)

    experiment = experiment_class(**kwargs)
    experiment.run_experiment(name=args.name, version=args.version, output_path=args.output, resume_checkpoint=args.resume_checkpoint, gpus=args.gpus, nodes=args.nodes)

def parse_other_args(other_args):
    kwargs = {}
    for arg in other_args:
        parts = arg.split("=")
        k = parts[0]
        if len(parts) == 1:
            v = True
        else:
            v = "=".join(parts[1:])
            if v.startswith('"') or v.startswith("'"):
                v = v[1:-2]
            elif v == "True":
                v = True
            elif v == "False":
                v = False
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
        kwargs[k] = v
    return kwargs


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
                save_dir=output_path, version=version, name=name,
                log_graph=hasattr(self, "example_input_array"),
                default_hp_metric=False
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
                print(f"Resume Checkpoint: {resume_checkpoint}")
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
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        """
        Create a validation dataloader.

        The default implementation wraps self.val_data in a Dataloader.
        """
        shuffle = True
        if isinstance(self.val_data, IterableDataset):
            shuffle = False
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers)

    def log_resources(self, gpus_separately=False):
        """
        Log the cpu, ram and gpu usage.
        """
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / 1000000000
        self.log("sys/SYS_CPU (%)", cpu)
        self.log("sys/SYS_RAM (GB)", ram)
        total_gpu_load = 0
        total_gpu_mem = 0
        for gpu in GPUtil.getGPUs():
            total_gpu_load += gpu.load
            total_gpu_mem += gpu.memoryUsed
            if gpus_separately:
                self.log("sys/GPU_UTIL_{}".format(gpu.id), gpu.load)
                self.log("sys/GPU_MEM_{}".format(gpu.id), gpu.memoryUtil)
        self.log("sys/GPU_UTIL (%)", total_gpu_load)
        self.log("sys/GPU_MEM (GB)", total_gpu_mem / 1000)

    def log_fps(self):
        """
        Log the FPS that is achieved.
        """
        if hasattr(self, "_iter_time"):
            elapsed = time() - self._iter_time
            fps = self.hparams.batch_size / elapsed
            self.log("sys/FPS", fps)
        self._iter_time = time()

    def train(self, mode=True):
        """
        Set the experiment to training mode and val mode.

        This is done automatically. You will not need this usually.
        """
        if self.logger is not None and hasattr(self.logger, "set_mode"):
            self.logger.set_mode("train" if mode else "val")
        super().train(mode)
