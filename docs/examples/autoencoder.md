[Back to Overview](../README.md)



# Example: Autoencoder

> A sample on how to write a fully custom experiment.

See the source code [examples/autoencoder.py](../../examples/autoencoder.py)


---
---
## *class* **AutoEncoderExperiment**(Experiment)

*(no documentation found)*

---
### *def* **forward**(*self*, x)

In lightning, forward defines the prediction/inference actions.


---
### *def* **step**(*self*, feature, target, batch_idx)

Step defined the train/val loop. It is independent of forward.


---
### *def* **prepare_data**(*self*)

Prepare the data once (no state allowed due to multi-gpu/node setup.)


---
### *def* **load_data**(*self*, stage=None) -> Tuple[Any, Any]

Load your datasets.

* **returns**: Tuple of train, val dataset.


---
### *def* **configure_optimizers**(*self*)

Create an optimizer to your liking.

* **returns**: The torch optimizer used for training.


---
### *def* **main**()

Manually run the training.


