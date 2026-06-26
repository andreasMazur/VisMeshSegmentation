from geoconv_examples.mpi_faust.pytorch.model import Imcnn

from torch import nn
from torch.nn import functional as F

class SegImcnn(nn.Module):
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

        self.modules = nn.ModuleList(
            [
                self.model.normalize,
                self.model.downsize_dense,
                self.model.downsize_activation,
                self.model.downsize_bn,
                self.model.output_dense
            ]
        )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.model.normalize.mean = self.model.normalize.mean.to(*args, **kwargs)
        self.model.normalize.var = self.model.normalize.var.to(*args, **kwargs) 
        return self

    def forward(self, inputs):
        signal, bc = inputs
        signal = self.model.normalize(signal)
        signal = self.model.downsize_dense(signal)
        signal = self.model.downsize_activation(signal)
        signal = self.model.downsize_bn(signal)

        for idx in range(len(self.model.output_dims)):
            signal = self.model.do_layers[idx](signal)
            signal = self.model.isc_layers[idx]([signal, bc])
            signal = self.model.amp_layers[idx](signal)
            signal = self.model.bn_layers[idx](signal)
        logits = self.model.output_dense(signal)

        return F.log_softmax(logits, dim=-1), signal