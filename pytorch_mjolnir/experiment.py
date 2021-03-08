"""doc
# pytorch_mjolnir.Experiment

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
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from time import time
from datetime import datetime
from torch.utils.data.dataset import IterableDataset
from deeptech.data.dataset import Dataset
from deeptech.core.definitions import SPLIT_TRAIN, SPLIT_VAL
from pytorch_lightning.utilities.cloud_io import load as pl_load

from pytorch_mjolnir.utils.tensorboard import TensorBoardLogger


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
    parser.add_argument('--evaluate_checkpoint', type=str, required=False, default=None, help='A specific checkpoint to evaluate.')
    args, other_args = parser.parse_known_args()

    kwargs = parse_other_args(other_args)

    experiment = experiment_class(**kwargs)
    if args.evaluate_checkpoint is None:
        experiment.run_experiment(name=args.name, version=args.version, output_path=args.output, resume_checkpoint=args.resume_checkpoint, gpus=args.gpus, nodes=args.nodes)
    else:
        experiment.evaluate_experiment(name=args.name, version=args.version, output_path=args.output, evaluate_checkpoint=args.evaluate_checkpoint, gpus=args.gpus, nodes=args.nodes)

def parse_other_args(other_args):
    kwargs = {}
    for arg in other_args:
        parts = arg.split("=")
        k = parts[0]
        if k.startswith("--"):
            k = k[2:]
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
        self.output_path = os.path.join(output_path, name, version)
        self.testing = False
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
            resume_from_checkpoint=resume_checkpoint,
            accelerator="ddp" if gpus > 1 else None
        )
        trainer.fit(self)

    def evaluate_experiment(self, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), evaluate_checkpoint=None):
        """
        Evaluate the experiment.

        :param name: The name of the family of experiments you are conducting.
        :param gpus: The number of gpus used for training.
        :param nodes: The number of nodes used for training.
        :param version: The name for the specific run of the experiment in the family (defaults to a timestamp).
        :param output_path: The path where to store the outputs of the experiment (defaults to the current working directory).
        :param evaluate_checkpoint: The path to the checkpoint that should be loaded (defaults to None).
        """
        if version is None:
            version = _generate_version()
        if evaluate_checkpoint is None:
            raise RuntimeError("No checkpoint provided for evaluation, you must provide one.")
        self.output_path = os.path.join(output_path, name, version)
        if evaluate_checkpoint == "last":
            checkpoint_path = self._find_checkpoint(name, version, output_path)
        else:
            checkpoint_path = os.path.join(self.output_path, evaluate_checkpoint)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Checkpoint does not exist: {str(checkpoint_path)}")
        self.testing = True
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
            accelerator="ddp" if gpus > 1 else None
        )
        ckpt = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt['state_dict'])
        trainer.test(self)

    def _find_checkpoint(self, name, version, output_path):
        resume_checkpoint = None
        checkpoint_folder = os.path.join(output_path, name, version, "checkpoints")
        if os.path.exists(checkpoint_folder):
            checkpoints = sorted(os.listdir(checkpoint_folder))
            if len(checkpoints) > 0:
                resume_checkpoint = os.path.join(checkpoint_folder, checkpoints[-1])
                print(f"Using Checkpoint: {resume_checkpoint}")
        return resume_checkpoint

    def get_cached_dataset(self, split):
        cache_path = getattr(self.hparams, "cache_path", None)
        if cache_path is not None:
            full_cache_path = cache_path + "_" + split
            if os.path.exists(full_cache_path):
                dataset = Dataset.from_disk(full_cache_path, split)
                assert len(dataset) > 0
                return dataset
        dataset = self.get_dataset(split)
        if cache_path is not None:
            full_cache_path = cache_path + "_" + split
            cache_exists = os.path.exists(full_cache_path)
            dataset.init_caching(full_cache_path)
            if not cache_exists:
                self.cache_data(dataset, split)
        assert len(dataset) > 0
        return dataset

    def cache_data(self, dataset, name):
        dataloader = DataLoader(dataset, num_workers=getattr(self.hparams, "num_workers", None))
        for _ in tqdm(dataloader, desc=f"Caching {name}"):
            pass

    def get_dataset(self, split) -> Tuple[Any, Any]:
        """
        **ABSTRACT:** Load the data for a given split.

        :return: A dataset.
        """
        raise NotImplementedError("Must be implemented by inheriting classes.")

    def prepare_data(self):
        # Prepare the data once (no state allowed due to multi-gpu/node setup.)
        if not self.testing:
            self.get_cached_dataset(SPLIT_TRAIN)
            self.get_cached_dataset(SPLIT_VAL)
            assert not getattr(self.hparams, "data_only_prepare", False)

    def load_data(self, stage=None) -> Tuple[Any, Any]:
        """
        **ABSTRACT:** Load the data for training and validation.

        :return: A tuple of the train and val dataset.
        """
        if not self.testing:
            return self.get_cached_dataset(SPLIT_TRAIN), self.get_cached_dataset(SPLIT_VAL)
        else:
            return None, None

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
            if self.testing:
                self.logger.set_mode("test")
            else:
                self.logger.set_mode("train" if mode else "val")
        super().train(mode)
