import json
import logging
from math import sqrt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.pyplot import MultipleLocator
from scipy.cluster.hierarchy import dendrogram
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

from .log_utils import Log

LOGGER = Log().init_logger(logger_name=__name__)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.ERROR)
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

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

def plot_heatmap(preds: np.ndarray, labels: np.ndarray, num_classes: int, pic_title: str, save_fig_path: Path, can_show_conf_mat: bool = False):
    """Plots the heatmap of the confusion matrix."""
    assert num_classes % 2 != 0, "the number of classes should be odd"
    classes_range = range(-1*(num_classes//2), (num_classes//2)+1)
    total_data_confusion_matrix = pd.DataFrame(confusion_matrix(labels.reshape(-1), preds.reshape(-1), labels=range(num_classes)), columns=classes_range, index=classes_range)
    plt.figure(figsize = (17, 17))
    plt.rcParams.update({'font.size': 44})
    ax = plt.gca()
    heatmap(total_data_confusion_matrix, annot=True, ax=ax, fmt='g')
    ax.set(xlabel="Prediction", ylabel="Ground Truth", title=pic_title)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    if can_show_conf_mat:
        total_data_confusion_matrix.index.name = 'Ground Truth'
        total_data_confusion_matrix.columns.name = 'Prediction'
        LOGGER.info(f"\nconfusion_matrix:\n{total_data_confusion_matrix}")
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
                      {"sub_title": "model structure",
                       "data": str(model_struct)}]

    # figrue settings
    line_style = {"linewidth": 2, "alpha": 0.5}
    axvline_style = {"color": 'k', "linewidth": 5, "linestyle": '--', "alpha": 0.3}
    fig, axes = plt.subplot_mosaic("""
                                   ab
                                   cc
                                   """,
                                   figsize=(30, 20), gridspec_kw={'hspace': 0.2, 'wspace': 0.3})
    fig.suptitle(main_title, fontsize=30)

    try:
        for ax, data_plot in zip(axes.values(), data_info_dict):
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
        LOGGER.error(f"Encounter error when draw figure of {data_plot['sub_title']}")
        raise e

    fig.tight_layout(rect=(0, 0, 0, 0))
    plt.show()
    plt.close()

def plot_cluster_labels_distribution(trained_cluster_model: sklearn.base.ClusterMixin, cluster_name: str, fig_title: str, save_dir: Path = None):
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.bar(np.unique(trained_cluster_model.labels_, return_counts=True)[0], np.unique(trained_cluster_model.labels_, return_counts=True)[1])
    plt.grid()
    plt.ylabel("samples in cluster")
    plt.xlabel("cluster label")
    plt.title(f"{cluster_name}\n {fig_title}")
    if save_dir is not None:
        plt.savefig(save_dir/f"{cluster_name}_{fig_title}.png")
    plt.show()  # findout elbow point
    plt.close()
    LOGGER.info(f"cluster of each point distribution: {np.unique(trained_cluster_model.labels_, return_counts=True)}")


def plot_dendrogram(trained_cluster_model: sklearn.base.ClusterMixin, save_dir: Path, **kwargs):
    """
    Create linkage matrix and then plot the dendrogram
    """
    assert hasattr(trained_cluster_model, "children_"), "trained_cluster_model must have children_ attribute"
    # create the counts of samples under each node
    counts = np.zeros(trained_cluster_model.children_.shape[0])
    n_samples = len(trained_cluster_model.labels_)
    for i, merge in enumerate(trained_cluster_model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [trained_cluster_model.children_, trained_cluster_model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    ax = plt.gca()
    kwargs["ax"] = ax
    dendrogram(linkage_matrix, **kwargs)
    if save_dir is not None:
        plt.savefig(save_dir/"dendrogram.png")
    plt.show()
    plt.close()


def plot_silhouette(ax: mpl.axes._axes.Axes, n_clusters: int, data: np.ndarray, silhouette_avg: float, sample_silhouette_values: np.ndarray, each_sample_cluster_labels: np.ndarray):
    """Plot silhouette coefficient for each sample"""
    cmp = mpl.colormaps['rainbow']
    ax.set_xlim([-1, 1])  # The silhouette coefficient can range from -1, 1
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])  # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[each_sample_cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cmp(float(i)/n_clusters)
        ax.fill_betweenx(y=np.arange(y_lower, y_upper), x1=0, x2=ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"cluster_{i}")  # Label the silhouette plots with their cluster numbers at the middle
        y_lower = y_upper + 10  # Compute the new y_lower for next plot, and 10 for the 0 samples
    ax.set_title(f"For n_clusters={n_clusters}, The average silhouette_score is: {silhouette_avg}", fontsize=14)
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")  # The vertical line for average silhouette score of all the values
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks(np.linspace(-1, 1, 11))


def plot_cluster_scatters(ax: mpl.axes._axes.Axes, n_clusters: int, data: np.ndarray, each_sample_cluster_labels: np.ndarray, centers: np.ndarray):
    """Plot the cluster scatters"""
    cmp = mpl.colormaps['rainbow']
    color_each_data = cmp(each_sample_cluster_labels.astype(float)/n_clusters)
    if data.shape[1] > 1:
        ax.scatter(x=data[:, 0], y=data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=color_each_data, edgecolor="k")
        ax.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")  # Draw white circles at cluster centers
        for i, c in enumerate(centers):
            ax.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    else:
        ax.scatter(x=data[:, 0], y=np.zeros_like(data[:, 0]), marker=".", s=30, lw=0, alpha=0.7, c=color_each_data, edgecolor="k")
        ax.scatter(x=centers[:, 0], y=np.zeros_like(centers[:, 0]), marker="o", c="white", alpha=1, s=200, edgecolor="k")
        for i, c in enumerate(centers):
            ax.scatter(x=c[0], y=np.zeros_like(c[0]), marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax.set_title("The visualization of the clustered data.", fontsize=14)
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")


def plot_cluster_scores(ax: mpl.axes._axes.Axes, silhouette_avg: float, db_score: float, ch_score: float):
    """Plot the cluster scores"""
    scores_dict = {"silhouette_avg": silhouette_avg, "db_score": db_score, "ch_score": ch_score}
    cmp = mpl.colormaps['rainbow']
    color_each_score_type = cmp(len(scores_dict.keys()))
    ax.set_title(f"The silhouette_avg is: {silhouette_avg}\nThe davies_bouldin_score is: {db_score}\nThe calinski_harabasz_score is: {ch_score}", fontsize=14)
    bars = ax.bar(scores_dict.keys(), scores_dict.values(), color=color_each_score_type)
    ax.bar_label(bars, fmt='%.8f', label_type='center', fontsize=14)


def plot_table(ax: mpl.axes._axes.Axes, df: pd.DataFrame):
    """Plot the table"""
    n_rows, n_cols = df.shape
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, colColours=["palegreen"]*n_cols, rowColours=["palegreen"]*n_rows, bbox=[0, 0, 1, 1])
    # modify table
    table.set_fontsize(14)
    table.scale(1, 1.5)
    ax.axis('off')

def plot_cluster_info(data: np.ndarray, each_sample_cluster_labels: np.ndarray, cluster_centers: np.ndarray,
                      n_clusters: int, linkage: str, cluster_metric: str,
                      sample_silhouette_values: np.ndarray, silhouette_avg: float, db_score: float, ch_score: float,
                      clusters_info_df: pd.DataFrame):
    """Plot the cluster info"""
    mosaic_str = (f"ab\n"
                  f"cc")
    df_row_len_20_quotient = (clusters_info_df.shape[0]-1)//20  # 20 is the number of rows of each column, and -2 is for excluding the title row and the last row
    fig_row = df_row_len_20_quotient+2
    varied_figsize = (32, 5+10*(fig_row))
    for i in range(df_row_len_20_quotient+1):
        mosaic_str += f"\n{chr(ord('d')+i)}{chr(ord('d')+i)}"
    fig, axes = plt.subplot_mosaic(mosaic_str, figsize=varied_figsize, gridspec_kw={'hspace': 0.1, 'wspace': 0.2})
    fig.suptitle(f"Silhouette analysis for Hierarchy clustering on sample data with n_clusters={n_clusters} linkage={linkage} metric={cluster_metric}", fontsize=20, fontweight="bold", y=0.91)
    plot_silhouette(ax=axes['a'], n_clusters=n_clusters, data=data, silhouette_avg=silhouette_avg, sample_silhouette_values=sample_silhouette_values, each_sample_cluster_labels=each_sample_cluster_labels)
    plot_cluster_scores(ax=axes['b'], silhouette_avg=silhouette_avg, db_score=db_score, ch_score=ch_score)
    plot_cluster_scatters(ax=axes['c'], n_clusters=n_clusters, data=data, each_sample_cluster_labels=each_sample_cluster_labels, centers=cluster_centers)
    table_axes = list(axes.values())[ord(max(mosaic_str))-97-df_row_len_20_quotient:]
    splitted_dfs = [clusters_info_df.iloc[i*20:(i+1)*20, :] for i in range(df_row_len_20_quotient+1)]
    for ax, splitted_df in zip(table_axes, splitted_dfs):
        display_df = splitted_df.iloc[::, :10]  # display the first 10 columns
        plot_table(ax=ax, df=display_df)
    plt.show()
    plt.close()


def plot_tr_val_labels_pie(tr_labels: np.ndarray, val_labels: np.ndarray, tr_labels_freq_counts: np.ndarray, val_labels_freq_counts: np.ndarray, axes: mpl.axes._axes.Axes = None):
    """Plot the pie chart of the train and validation labels"""
    colors_labels_map = {"-1.0": "lime", "0.0": "darkorange", "1.0": "dodgerblue"}
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        axes[0].pie(tr_labels_freq_counts, labels=tr_labels, autopct='%1.1f%%', textprops={'fontsize': 24}, colors=[colors_labels_map[str(label)] for label in tr_labels])
        axes[0].set_title("Train", fontsize=32)
        axes[1].pie(val_labels_freq_counts, labels=val_labels, autopct='%1.1f%%', textprops={'fontsize': 24}, colors=[colors_labels_map[str(label)] for label in val_labels])
        axes[1].set_title("Validation", fontsize=32)
        plt.show()
        plt.close()
    else:
        axes[0, 0].pie(tr_labels_freq_counts, labels=tr_labels, autopct='%1.1f%%', textprops={'fontsize': 24}, colors=[colors_labels_map[str(label)] for label in tr_labels])
        axes[0, 0].set_title("Train", fontsize=32)
        axes[0, 1].pie(val_labels_freq_counts, labels=val_labels, autopct='%1.1f%%', textprops={'fontsize': 24}, colors=[colors_labels_map[str(label)] for label in val_labels])
        axes[0, 1].set_title("Validation", fontsize=32)


def plot_corr_distribution(corr_df: pd.DataFrame, axes: mpl.axes._axes.Axes = None):
    """Plot the correlation distribution"""
    all_item_pair_corrs = np.hstack(corr_df.values)
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        axes[0].hist(all_item_pair_corrs, bins=20)
        axes[0].xaxis.set_tick_params(labelsize=18)
        axes[1].boxplot(all_item_pair_corrs)
        axes[1].yaxis.set_tick_params(labelsize=18)
        plt.show()
        plt.close()
    else:
        axes[1, 0].hist(all_item_pair_corrs, bins=20)
        axes[1, 0].xaxis.set_tick_params(labelsize=18)
        axes[1, 1].boxplot(all_item_pair_corrs)
        axes[1, 1].yaxis.set_tick_params(labelsize=18)
