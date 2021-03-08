[Back to Overview](../README.md)



# pytorch_mjolnir.Experiment

> A lightning module that runs an experiment in a managed way.

There is first the Experiment base class from wich all experiments must inherit (directly or indirectly).


---
### *def* **run**(experiment_class)

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


---
### *def* **parse_other_args**(other_args)

*(no documentation found)*

---
---
## *class* **Experiment**(pl.LightningModule)

An experiment base class.

All experiments must inherit from this.

```python
from pytorch_mjolnir import Experiment
class MyExperiment(Experiment):
[...]
```


---
### *def* **run_experiment**(*self*, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), resume_checkpoint=None)

Run the experiment.

* **name**: The name of the family of experiments you are conducting.
* **gpus**: The number of gpus used for training.
* **nodes**: The number of nodes used for training.
* **version**: The name for the specific run of the experiment in the family (defaults to a timestamp).
* **output_path**: The path where to store the outputs of the experiment (defaults to the current working directory).
* **resume_checkpoint**: The path to the checkpoint that should be resumed (defaults to None).
In case of None this searches for a checkpoint in {output_path}/{name}/{version}/checkpoints and resumes it.
Without defining a version this means no checkpoint can be found as there will not exist a  matching folder.


---
### *def* **evaluate_experiment**(*self*, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), evaluate_checkpoint=None)

Evaluate the experiment.

* **name**: The name of the family of experiments you are conducting.
* **gpus**: The number of gpus used for training.
* **nodes**: The number of nodes used for training.
* **version**: The name for the specific run of the experiment in the family (defaults to a timestamp).
* **output_path**: The path where to store the outputs of the experiment (defaults to the current working directory).
* **evaluate_checkpoint**: The path to the checkpoint that should be loaded (defaults to None).


---
### *def* **get_cached_dataset**(*self*, split)

*(no documentation found)*

---
### *def* **cache_data**(*self*, dataset, name)

*(no documentation found)*

---
### *def* **get_dataset**(*self*, split) -> Tuple[Any, Any]

**ABSTRACT:** Load the data for a given split.

* **returns**: A dataset.


---
### *def* **prepare_data**(*self*)

*(no documentation found)*

---
### *def* **load_data**(*self*, stage=None) -> Tuple[Any, Any]

**ABSTRACT:** Load the data for training and validation.

* **returns**: A tuple of the train and val dataset.


---
### *def* **training_step**(*self*, batch, batch_idx)

Executes a training step.

By default this calls the step function.
* **batch**: A batch of training data received from the train loader.
* **batch_idx**: The index of the batch.


---
### *def* **validation_step**(*self*, batch, batch_idx)

Executes a validation step.

By default this calls the step function.
* **batch**: A batch of val data received from the val loader.
* **batch_idx**: The index of the batch.


---
### *def* **setup**(*self*, stage=None)

This function is for setting up the training.

The default implementation calls the load_data function and
stores the result in self.train_data and self.val_data.
(It is called once per process.)


---
### *def* **train_dataloader**(*self*)

Create a training dataloader.

The default implementation wraps self.train_data in a Dataloader.


---
### *def* **val_dataloader**(*self*)

Create a validation dataloader.

The default implementation wraps self.val_data in a Dataloader.


---
### *def* **log_resources**(*self*, gpus_separately=False)

Log the cpu, ram and gpu usage.


---
### *def* **log_fps**(*self*)

Log the FPS that is achieved.


---
### *def* **train**(*self*, mode=True)

Set the experiment to training mode and val mode.

This is done automatically. You will not need this usually.


