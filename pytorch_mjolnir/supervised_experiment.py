"""doc
# pytorch_mjolnir.SupervisedExperiment

> An implementation of an experiment for supervised training.

You simply set a model and a loss as attributes in the constructor and the experiment takes care of the rest.
"""

from pytorch_mjolnir.experiment import Experiment


class SupervisedExperiment(Experiment):
    """
    A supervised experiment implements forward and step by using the model and loss variable.

    In your constructor simply define:
    ```
    def __init__(self, learning_rate=1e-3, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model()
        self.loss = Loss()
    ```
    """
    def forward(self, *args, **kwargs):
        """
        Proxy to self.model.
        
        Arguments get passed unchanged.
        """
        return self.model(*args, **kwargs)

    def step(self, feature, target, batch_idx):
        """
        Implementation of a supervised training step.

        The output of the model will be directly given to the loss without modification.

        :param feature: A namedtuple from the dataloader that will be given to the forward as ordered parameters.
        :param target: A namedtuple from the dataloader that will be given to the loss.
        :return: The loss.
        """
        prediction = self(*feature)
        self.loss.log = self.log
        loss = self.loss(prediction, target)
        if self.training:
            self.log('loss/train', loss)
        else:
            self.log('loss/val', loss)
        return loss
