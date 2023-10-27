import json
import logging
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s.%(funcName)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)

def set_plot_log_data(log_path: Path):
    """Sets the data for plotting."""
    with open(log_path, "r") as f:
        log = json.load(f)

    tr_preds_history = log['tr_preds_history']
    tr_labels_history = log['tr_labels_history']
    val_preds_history = log['val_preds_history']
    val_labels_history = log['val_labels_history']
    graph_adj_mat_size = len(tr_preds_history[0][0])
    num_epochs = len(tr_preds_history)
    num_obs_epochs = 5
    obs_epochs = np.linspace(0, num_epochs-1, num_obs_epochs, dtype="int")
    obs_best_val_epoch = np.argmax(np.array(log['val_edge_acc_history']))
    obs_batch_idx = 10
    best_epoch_val_preds = np.array(val_preds_history[obs_best_val_epoch])
    best_epoch_val_labels = np.array(val_labels_history[obs_best_val_epoch])

    if sqrt(len(tr_preds_history[0][0])).is_integer() and len(tr_preds_history[0][0]) != 1:
        num_nodes = int(sqrt(len(tr_preds_history[0][0])))
        is_square_graph = True
    else:
        num_nodes_minus_one = 1
        while (num_nodes_minus_one**2 + num_nodes_minus_one)/2 != len(tr_preds_history[0][0]):  # arithmetic progression sum formula
            num_nodes_minus_one += 1
        num_nodes = num_nodes_minus_one+1
        is_square_graph = False
    assert is_square_graph == (graph_adj_mat_size == num_nodes**2), "when the graph is square graph, the size of graph should be num_nodes**2"
    assert len(tr_preds_history) == len(tr_labels_history) and len(tr_preds_history) == len(val_preds_history) and len(tr_preds_history) == len(val_labels_history), "length of {tr_preds_history, tr_labels_history, val_preds_history, val_labels_history} should be equal to num_epochs"
    return_dict = {key: value for key, value in locals().items() if key not in ['f', 'log', 'log_path', 'num_epochs', 'obs_best_val_epoch']}

    return return_dict

def plot_heatmap(preds: np.ndarray, labels: np.ndarray, save_fig_path: Path = None, can_show_conf_mat: bool = False):
    total_data_confusion_matrix = pd.DataFrame(confusion_matrix(labels.reshape(-1), preds.reshape(-1), labels=[0, 1, 2]), columns=range(-1, 2), index=range(-1, 2))
    plt.figure(figsize = (10, 10))
    plt.rcParams.update({'font.size': 44})
    ax = plt.gca()
    heatmap(total_data_confusion_matrix, annot=True, ax=ax, fmt='g')
    ax.set(xlabel="Prediction", ylabel="Ground Truth", title="val")
    if can_show_conf_mat:
        logger.info(f"confusion_matrix:\n{total_data_confusion_matrix}")
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()
    plt.close()
