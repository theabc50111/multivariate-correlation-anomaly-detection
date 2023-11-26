import inspect
import logging
from math import sqrt

import numpy as np
import torch
import torch_geometric
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import unbatch, unbatch_edge_index

from .log_utils import Log

LOGGER = Log().init_logger(logger_name=__name__)

# set devide of pytorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
torch.set_default_dtype(torch.float64)


class TolEdgeAccuracyLoss(torch.nn.Module):
    """ 
    This loss function is used to compute the edge accuracy of the prediction.
    """
    def __init__(self):
        super(TolEdgeAccuracyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, atol: float = 0.05) -> torch.Tensor:
        raise NotImplementedError("This function is not implemented yet.")
        ###edge_acc = torch.isclose(input, target, atol=atol, rtol=0).to(torch.float64).mean()
        ###edge_acc.requires_grad = True
        ###loss = 1 - edge_acc
        ###return loss

class CustomIndicesCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes: int, selected_indices: list, weight: torch.Tensor):
        super(CustomIndicesCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.selected_indices = selected_indices
        self.weight = weight


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, input_feature_size = input.shape
        assert num_classes == self.num_classes, "The number of classes in the input tensor is not equal to the number of classes in the model."
        assert input_feature_size > 1, "The input data feature size must be greater than 1."
        selected_input = input[::, ::, self.selected_indices]
        selected_target = target[::, self.selected_indices]
        loss = CrossEntropyLoss(weight=self.weight)(input=selected_input, target=selected_target)

        return loss


class CustomIndicesEdgeAccuracy(torch.nn.Module):
    def __init__(self, num_classes: int, selected_indices: list):
        super(CustomIndicesEdgeAccuracy, self).__init__()
        self.num_classes = num_classes
        self.selected_indices = selected_indices

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, input_feature_size = input.shape
        assert num_classes == self.num_classes, "The number of classes in the input tensor is not equal to the number of classes in the model."
        assert input_feature_size > 1, "The input data feature size must be greater than 1."
        selected_input = input[::, ::, self.selected_indices]
        selected_target = target[::, self.selected_indices]
        selected_preds = torch.argmax(selected_input, dim=1)
        edge_acc = (selected_preds == selected_target).to(torch.float64).mean()

        return edge_acc
