[Back to Overview](../README.md)



# pytorch_mjolnir.Experiment / mjolnir_experiment

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
### *def* **validation_epoch_end**(*self*, val_step_outputs)

This function is called after all training steps.

It accumulates the loss into a val_loss which is logged in the end.


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


