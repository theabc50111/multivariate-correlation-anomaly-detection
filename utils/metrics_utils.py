import inspect
import logging
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import unbatch, unbatch_edge_index

from .log_utils import Log

LOGGER = Log().init_logger(logger_name=__name__)
DF_LOGGER = Log().init_logger(logger_name="df_logger")

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
        raise NotImplementedError
        ###edge_acc = torch.isclose(input, target, atol=atol, rtol=0).to(torch.float64).mean()
        ###edge_acc.requires_grad = True
        ###loss = 1 - edge_acc
        ###return loss


class TolEdgeAccuracy(torch.nn.Module):
    """ 
    This metric function is used to compute the edge accuracy of the prediction.
    """
    def __init__(self, atol: float = 0.05):
        super(TolEdgeAccuracy, self).__init__()
        self.atol = atol

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge_acc = torch.isclose(input, target, atol=self.atol, rtol=0).to(torch.float64).mean()
        return edge_acc


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


def report_preds_correctness(model_input_df: pd.DataFrame, tr_val_tt_len_list: list, inference_data: dict, preds: np.ndarray, labels: np.ndarray, data_sp_mode: str, report_save_path: Path) -> None:
    """
    This function is used to report the false predictions of the model.
    """
    num_pairs, tol_time_steps = model_input_df.shape
    assert num_pairs == inference_data["model_input"].shape[0] == inference_data["target"].shape[0], "The number of pairs in the model input dataframe must be equal to the number of pairs in the inference data."
    assert len(tr_val_tt_len_list) == 3, "The length of the `tr_val_tt_list` list must be 3."
    assert sum(tr_val_tt_len_list) == tol_time_steps, "The sum of the `tr_val_tt_list` list must be equal to the total number of time steps in the model input dataframe."
    assert inference_data["target"].shape[1] in tr_val_tt_len_list, "The number of time steps in the inference data target must be in the `tr_val_tt_list` list."
    model_input_dates = model_input_df.columns
    tr_dates_range = (model_input_dates[0:int(tr_val_tt_len_list[0])])
    val_dates_range = (model_input_dates[int(tr_val_tt_len_list[0]):int(tr_val_tt_len_list[0])+int(tr_val_tt_len_list[1])])
    test_dates_range = (model_input_dates[int(tr_val_tt_len_list[0])+int(tr_val_tt_len_list[1]):])
    assert len(tr_dates_range) == tr_val_tt_len_list[0] and len(val_dates_range) == tr_val_tt_len_list[1] and len(test_dates_range) == tr_val_tt_len_list[2], "The length of the dates range must be equal to the length of the `tr_val_tt_len` list."
    data_sp_dates = {"tr": tr_dates_range, "val": val_dates_range, "test": test_dates_range}[data_sp_mode]
    report_df = pd.DataFrame()
    for i, pair_name in enumerate(model_input_df.index):
        labels_each_pair = labels[:, i].astype("int64")
        preds_each_pair = preds[:, i].astype("int64")
        correct_preds_each_pair = (labels_each_pair == preds_each_pair)
        selected_dates = data_sp_dates[-1*len(preds_each_pair):]
        corr_coef_data = model_input_df.loc[pair_name, selected_dates].to_numpy()
        ori_labels = inference_data["target"][i, -1*len(preds_each_pair):].astype("int64")
        df_each_pair_idx = pd.MultiIndex.from_product([[pair_name], ["corr_coef", "ori_labels", "new_labels", "preds", "correct_pred"]], names=["pair_name", "data_category"])
        df_each_pair = pd.DataFrame([corr_coef_data, ori_labels, labels_each_pair, preds_each_pair, correct_preds_each_pair], columns=selected_dates, index=df_each_pair_idx)
        report_df = pd.concat([report_df, df_each_pair], axis=0)
    if report_save_path is not None:
        report_df.to_csv(report_save_path)

    LOGGER.info(f"num_pairs: {num_pairs}, data_sp_mode: {data_sp_mode}, data_sp_dates_range: `{data_sp_dates[0]} ~ {data_sp_dates[-1]}`, selected_dates_range: `{selected_dates[0]} - {selected_dates[-1]}`")
    DF_LOGGER.info(f"report_df:\n{report_df}")
    LOGGER.info("-"*50)

