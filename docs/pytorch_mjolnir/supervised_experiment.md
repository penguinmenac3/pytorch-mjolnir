[Back to Overview](../README.md)



# pytorch_mjolnir.SupervisedExperiment

> An implementation of an experiment for supervised training.

You simply set a model and a loss as attributes in the constructor and the experiment takes care of the rest.


---
---
## *class* **SupervisedExperiment**(Experiment)

A supervised experiment implements forward and step by using the model and loss variable.

In your constructor simply define:
```
def __init__(self, learning_rate=1e-3, batch_size=32):
super().__init__()
self.save_hyperparameters()
self.model = Model()
self.loss = Loss()
```


---
### *def* **forward**(*self*, *args, **kwargs)

Proxy to self.model.

Arguments get passed unchanged.


---
### *def* **step**(*self*, feature, target, batch_idx)

Implementation of a supervised training step.

The output of the model will be directly given to the loss without modification.

* **feature**: A namedtuple from the dataloader that will be given to the forward as ordered parameters.
* **target**: A namedtuple from the dataloader that will be given to the loss.
* **returns**: The loss.


