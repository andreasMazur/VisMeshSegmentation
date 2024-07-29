from geoconv_examples.mpi_faust.pytorch.model import Imcnn

from torch import nn


class SegImcnn(nn.Module):
    """A wrapper class for Imcnn models that learn to segment meshes."""
    def __init__(self, adapt_data, layer_conf=None):
        super().__init__()

        if layer_conf is None:
            layer_conf = [(96, 1)]

        self.model = Imcnn(
            signal_dim=3,  # Use 3D-coordinates as input
            kernel_size=(5, 8),
            adapt_data=adapt_data,
            layer_conf=layer_conf,
            variant="dirac",
            segmentation_classes=2,
            template_radius=0.544067211679114
        )

    def forward(self, x):
        """Forwards inputs through the model.

        Parameters
        ----------
        x: (torch.Tensor, torch.Tensor)
            The input to the model. First element is the mesh signal, second are the barycentric coordinates.
        """
        return self.model.forward(x)

    def train_loop(self, dataset, loss_fn, optimizer, verbose=True, epoch=None):
        """Training loop of the model.

        Parameters
        ----------
        dataset: torch.IterableDataset
            The dataset on which the IMCNN shall be trained.
        loss_fn: Callable
            The loss function to use during training.
        optimizer: torch.optim.Optimizer
            The optimizer for training.
        verbose: bool
            Whether to print intermediate training information during training.
        epoch: int
            The current epoch. Only used for printing training information.
        """
        return self.model.train_loop(dataset, loss_fn, optimizer, verbose=verbose, epoch=epoch)

    def validation_loop(self, dataset, loss_fn, verbose=True):
        """Validation loop of the model.

        Parameters
        ----------
        dataset: torch.IterableDataset
            The dataset on which the IMCNN shall be validated.
        loss_fn: Callable
            The loss function to use during training.
        verbose: bool
            Whether to print intermediate training information during training.
        """
        return self.model.validation_loop(dataset, loss_fn, verbose=verbose)
