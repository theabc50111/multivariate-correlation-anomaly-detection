import json
import logging
from math import sqrt
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s.%(funcName)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.ERROR)

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

def plot_gru_tr_process(main_title: str, model_struct: str, metrics_history: dict, best_epoch: int):
    data_info_dict = [{"sub_title": 'train loss_history & edge_acc_history',
                       "data": {'tr_loss_history': metrics_history['tr_loss_history'],
                                'tr_edge_acc_history': metrics_history['tr_edge_acc_history']},
                       "xticks": None,
                       "xlabel": "epochs",
                       "double_y": True},
                      {"sub_title": 'val  loss_history & edge_acc_history',
                       "data": {'val_loss_history': metrics_history['val_loss_history'],
                                'val_edge_acc_history': metrics_history['val_edge_acc_history']},
                       "xticks": None,
                       "xlabel": "epochs",
                       "double_y": True},
                      {"sub_title": f"model structure",
                       "data": str(model_struct)}]

    # figrue settings
    line_style = {"linewidth": 2, "alpha": 0.5}
    axvline_style = {"color": 'k', "linewidth": 5, "linestyle": '--', "alpha": 0.3}
    fig, axs = plt.subplot_mosaic("""
                                  ab
                                  cc
                                  """,
                                  figsize=(30, 20), gridspec_kw={'hspace': 0.2, 'wspace': 0.3})
    fig.suptitle(main_title, fontsize=30)

    try:
        for ax, data_plot in zip(axs.values(), data_info_dict):
            ax.set_title(data_plot["sub_title"], fontsize=30)
            ax.yaxis.offsetText.set_fontsize(18)
            ax.tick_params(axis='both', which='major', labelsize=24)
            if isinstance(data_plot["data"], dict) and data_plot.get("double_y"):
                for i, key in enumerate(data_plot["data"]):
                    if i == 0:
                        ax.plot(data_plot["data"][key], label=key, **line_style)
                        ax.set_ylabel(key, fontsize=24)
                        ax.legend(fontsize=18)
                    else:
                        new_ax = ax.twinx()
                        new_ax.plot(data_plot["data"][key], label=key, color='r')
                        new_ax.set_ylabel(key, color='r', fontsize=24)
                        new_ax.legend(fontsize=18)
                        new_ax.tick_params(axis='both', colors='r', which='major', labelsize=24)
            elif isinstance(data_plot["data"], dict):
                [ax.plot(data_plot["data"][key], label=key, **line_style) for key in data_plot["data"]]
                ax.legend(fontsize=18)
            elif isinstance(data_plot["data"], str):
                ax.annotate(text=f"{data_plot['data']}",
                            xy=(0.15, 0.5), bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 5},
                            fontsize=20, fontfamily='monospace', xycoords='axes fraction', va='center')
            else:
                ax.plot(data_plot["data"], **line_style)
            if pos_tuple := data_plot.get("axvline"):
                for x_pos in pos_tuple:
                    ax.axvline(x=x_pos, **axvline_style)
            if xlabel := data_plot.get("xlabel"):
                ax.set_xlabel(xlabel, fontsize=24)
            if t := data_plot.get("xticks"):
                ax.set_xticks(ticks=range(0, len(t["label"])*t["intv"], t["intv"]), labels=t["label"], rotation=45)
    except Exception as e:
        logging.error(f"Encounter error when draw figure of {data_plot['sub_title']}")
        raise e

    fig.tight_layout(rect=(0, 0, 0, 0))
    plt.show()
    plt.close()