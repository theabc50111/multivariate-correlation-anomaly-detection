# Function: clustering utilities
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import issymmetric
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             pairwise_distances, silhouette_samples,
                             silhouette_score)
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler

from .log_utils import Log
from .plot_utils import plot_cluster_info

LOGGER = Log().init_logger(logger_name=__name__)
DF_LOGGER = Log().init_logger(logger_name="df_logger")
warnings.simplefilter("ignore")


def convert_pairs_data_to_proximity_mat(item_pairs_ser: pd.Series, item_names: tuple, fill_diag_val: float) -> pd.DataFrame:
    """
    Convert item pairs data to proximity matrix
    item_pairs_ser: pd.Series, index is item pairs, value is proximity
    item_names: tuple, item names
    fill_diag_val: float, fill diagonal value
    """
    assert item_pairs_ser.index.str.split(" & ").tolist() == [[item_pair[0], item_pair[1]] for item_pair in combinations(item_names, 2)], "item_pairs_ser.index is not equal to combinations(item_names, 2)"
    num_items = len(item_names)
    proximity_mat = np.full((num_items, num_items), fill_value=np.nan)
    triu_idxs = np.triu_indices(num_items, k=1)
    tril_idxs = np.tril_indices(num_items, k=-1)
    proximity_mat[triu_idxs[0], triu_idxs[1]] = 0
    proximity_mat[tril_idxs[0], tril_idxs[1]] = 0
    proximity_mat[triu_idxs[0], triu_idxs[1]] = item_pairs_ser.values
    proximity_mat = proximity_mat+proximity_mat.T
    np.fill_diagonal(proximity_mat, val=1)
    assert ~np.isnan(proximity_mat).any(), "proximity_mat has nan value"
    assert np.allclose(proximity_mat, proximity_mat.T), "proximity_mat is not symmetric"
    proximity_df = pd.DataFrame(proximity_mat, columns=item_names, index=item_names)

    return proximity_df


def filter_proximity_mat(proximity_mat: pd.DataFrame, filter_mask: pd.DataFrame, tmp_clique_dir: Path) -> tuple[pd.DataFrame, list]:
    """
    Filter proximity matrix by filter mask
    proximity_mat: pd.DataFrame, proximity matrix
    filter_mask: pd.DataFrame, filter mask
    tmp_clique_dir: Path, temporary clique directory
    """
    assert proximity_mat.shape == filter_mask.shape, "proximity_mat.shape is not equal to filter_mask.shape"
    assert proximity_mat.index.tolist() == filter_mask.index.tolist() and proximity_mat.columns.tolist() == filter_mask.columns.tolist(), "proximity_mat.index or proximity_mat.columns is not equal to filter_mask.index or filter_mask.columns"
    tmp_clique_dir.mkdir(parents=True, exist_ok=True)
    ori_diag_val = np.diag(proximity_mat.values).copy()
    proximity_mat[~filter_mask] = 0
    G = nx.from_pandas_adjacency(proximity_mat)
    max_clique = []
    clique_counter = Counter()
    train_start_t = time()
    for i, clique in enumerate(nx.find_cliques(G)):
        clique = sorted(clique)
        LOGGER.debug(f"{i}th clique: {clique}")
        clique_counter[f"len_{len(clique)}_clique"] += 1
        if len(clique) <= 5:
            with open(tmp_clique_dir/"tmp_cliques.txt", "a") as f:
                f.write(f"{clique}\n")
        if len(clique) >= len(max_clique):
            if len(clique) > len(max_clique):
                max_clique = clique
            elif sum(1 for c1, c2 in zip(clique, max_clique) if c1 > c2):
                max_clique = clique
        now_t = time()
        if now_t - train_start_t > 10800:  # 30 minutes
            LOGGER.warning(f"clique search time out: {now_t - train_start_t} seconds")
            break
    top_3_len_cliq_cnt_key = sorted(clique_counter, key=lambda x: int(x.split('_')[1]), reverse=True)[:3]
    top_3_len_cliq = [f"number of {cnt_key}: {clique_counter[cnt_key]}" for cnt_key in top_3_len_cliq_cnt_key]
    top_3_freq_cliq_len = [f"number of {clique_len}: {num}" for clique_len, num in clique_counter.most_common(3)]
    LOGGER.info(f" : \ \ntotal cliques: {clique_counter.total()}, top 3 num_cliques by frequent of len: {top_3_freq_cliq_len} top 3 cliques by len: {top_3_len_cliq}")
    proximity_mat = proximity_mat.loc[max_clique, max_clique]
    np.fill_diagonal(proximity_mat.values, ori_diag_val)
    return proximity_mat, max_clique


def calc_pca(data: pd.DataFrame, n_samples: int, variance_thres: float, verbose: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate PCA
      data: pd.DataFrame, data
      n_samples: int, number of samples
      variance_thres: float, variance threshold
    """
    pca = PCA(n_components=None, whiten=True)
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(data)
    pca.fit(norm_data)
    assert pca.n_samples_ == n_samples, "pca.n_samples_ is not equal to n_samples, pca.n_samples_: {pca.n_samples_}, n_samples: {n_samples}"
    reducted_data = pca.transform(norm_data)
    pca_info_df = pd.DataFrame({"singular_values": pca.singular_values_, "variance_ratio": pca.explained_variance_ratio_,
                                "pri_components": [f"pc_{i}" for i in range(pca.n_components_)]}).set_index(["pri_components"])
    num_over_thres_pri_components = sum(pca.explained_variance_ratio_ > variance_thres)
    num_under_thres_pri_components = pca.n_components_ - num_over_thres_pri_components
    selected_reducted_data = reducted_data[::, :num_over_thres_pri_components]
    selected_reducted_df = pd.DataFrame(selected_reducted_data, columns=[f"pc_{i}" for i in range(num_over_thres_pri_components)], index=data.index)
    selected_pri_components = pca.components_[:num_over_thres_pri_components, ::]
    selected_pri_components_info_df = pca_info_df.iloc[:num_over_thres_pri_components, ::]
    sum_selected_pri_components_info_df = selected_pri_components_info_df.sum(axis=0).to_frame().T
    sum_selected_pri_components_info_df.index = ["sum"]
    selected_pri_components_info_df = pd.concat([selected_pri_components_info_df, sum_selected_pri_components_info_df], axis=0)

    LOGGER.info("="*80)
    LOGGER.info(f"pca_explanation_variance_thres:{variance_thres}, num_over_thres_pri_components:{num_over_thres_pri_components}, num_under_thres_pri_components:{num_under_thres_pri_components}")
    DF_LOGGER.info("==================== Principle Components info ====================")
    DF_LOGGER.info(selected_pri_components_info_df)
    DF_LOGGER.info("==================== selected_reducted_df ====================")
    DF_LOGGER.info(selected_reducted_df)

    if verbose == 1:
        pca_input_data_samples = data.index
        pca_input_data_featues = data.columns
        ori_logger_level = LOGGER.getEffectiveLevel()
        LOGGER.setLevel(10)
        LOGGER.debug("####################### check pca precessing #######################")
        LOGGER.debug(f"pca_input_data.shape:{data.shape}, len(pca_input_data_samples):{len(pca_input_data_samples)}, len(pca_input_data_featues):{len(pca_input_data_featues)}")
        LOGGER.debug(f"pca_input_data_samples[:3]:{pca_input_data_samples[:3]}")
        LOGGER.debug(f"pca_input_data_featues[:3]:{pca_input_data_featues[:3]}")
        LOGGER.debug("####################### check pca precessing #######################")
        LOGGER.setLevel(ori_logger_level)

    return selected_reducted_df, selected_pri_components


def calc_hrchy_cluster_given_n_clusters(n_clusters: int, data: pd.DataFrame, cluster_conds: dict):
    """
    Calculate hierarchical clustering
        n_clusters: int, number of clusters
        cluster_linkage: str, linkage
        cluster_metric: str, metric
        data: pd.DataFrame, data
        cluster_conds: dict, cluster conditions
    """
    cluster_linkage = cluster_conds["linkage"]
    cluster_metric = cluster_conds["cluster_metric"]
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=cluster_linkage, metric=cluster_metric, compute_distances=True)
    nc_clf = NearestCentroid()
    # calculate cluster info
    each_sample_cluster_labels = clusterer.fit_predict(data)
    cluster_labels, cluster_n_samples = np.unique(each_sample_cluster_labels, return_counts=True)
    nc_clf.fit(data, each_sample_cluster_labels)
    cluster_centers = nc_clf.centroids_
    assert clusterer.n_clusters_ == n_clusters and clusterer.n_features_in_ == cluster_conds["n_features"] and len(clusterer.labels_) == cluster_conds["n_samples"], "clusterer.n_clusters_ is not equal to n_clusters or clusterer.n_features_in_ is not equal to n_features or len(clusterer.labels_) is not equal to n_samples"
    assert cluster_centers.shape[0] == n_clusters and cluster_centers.shape[1] == cluster_conds["n_features"], "cluster_centers.shape is not equal to (n_clusters, n_features)"

    silhouette_avg = silhouette_score(data, each_sample_cluster_labels)  # The silhouette_score gives the average value for all the samples. This gives a perspective into the density and separation of the formed clusters
    db_score = davies_bouldin_score(data, each_sample_cluster_labels)
    ch_score = calinski_harabasz_score(data, each_sample_cluster_labels)

    sample_silhouette_values = silhouette_samples(data, each_sample_cluster_labels)  # Compute the silhouette scores for each sample
    pair_cluster_center_dist = pairwise_distances(cluster_centers)  # Compute the diatance between center of clusters
    dist_dict = {f"dist_to_cluster_{i}": pair_cluster_center_dist[::, i] for i in range(n_clusters)}
    cluster_data_dict = {"cluster_label": cluster_labels, "cluster_n_samples": cluster_n_samples,
                         "cluster_silhouette_avg": [sample_silhouette_values[each_sample_cluster_labels == i].mean() for i in range(n_clusters)],
                         "cluster_silhouette_min": [sample_silhouette_values[each_sample_cluster_labels == i].min() for i in range(n_clusters)],
                         "cluster_silhouette_max": [sample_silhouette_values[each_sample_cluster_labels == i].max() for i in range(n_clusters)]}
    cluster_data_dict.update(dist_dict)
    clusters_info_df = pd.DataFrame(cluster_data_dict)

    return each_sample_cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values, db_score, ch_score, clusters_info_df


def obs_various_n_clusters_hrchy_cluster(data: pd.DataFrame, cluster_conds: dict, save_fig_path: Path = None, can_plot_each_cluster_info: bool = False):
    """
    Observe various n_clusters hierarchical clustering
    """
    n_clusters_list = cluster_conds["n_clusters_list"]
    linkage = cluster_conds["linkage"]
    cluster_metric = cluster_conds["cluster_metric"]
    assert data.shape[0] == cluster_conds["n_samples"] and data.shape[1] == cluster_conds["n_features"], "data.shape is not equal to (n_samples, n_features)"
    various_n_clusters_model_info_df = pd.DataFrame()
    for n_clusters in n_clusters_list:
        hrchy_ret = calc_hrchy_cluster_given_n_clusters(n_clusters=n_clusters, data=data, cluster_conds=cluster_conds)
        each_sample_cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values, db_score, ch_score, clusters_info_df = hrchy_ret
        various_n_clusters_model_info_df = pd.concat([various_n_clusters_model_info_df, pd.DataFrame({"n_clusters": n_clusters, "cluster_linkage": linkage, "cluster_metric": cluster_metric, "silhouette_avg": silhouette_avg, "db_score": db_score, "ch_score": ch_score}, index=[0])], axis=0)
        if can_plot_each_cluster_info:
            LOGGER.info(f"Plotting Hierarchical Clustering with n_clusters={n_clusters} linkage={linkage} cluster_metric={cluster_metric}")
            plot_cluster_info(data=data.values, each_sample_cluster_labels=each_sample_cluster_labels, cluster_centers=cluster_centers,
                              n_clusters=n_clusters, linkage=linkage, cluster_metric=cluster_metric,
                              sample_silhouette_values=sample_silhouette_values, silhouette_avg=silhouette_avg, db_score=db_score, ch_score=ch_score,
                              clusters_info_df=clusters_info_df)
    scaler = MinMaxScaler()
    cluster_score_df = various_n_clusters_model_info_df.loc[::, ["n_clusters", "silhouette_avg", "db_score", "ch_score"]]
    cluster_score_df.loc[::, ["silhouette_avg", "db_score", "ch_score"]] = scaler.fit_transform(cluster_score_df.loc[::, ["silhouette_avg", "db_score", "ch_score"]])
    cluster_score_df.loc[::, "reverse_db_score"] = 1 - cluster_score_df.loc[::, "db_score"]
    cluster_score_df.loc[::, "synthetic_score"] = cluster_score_df.loc[::, ["silhouette_avg", "reverse_db_score", "ch_score"]].sum(axis=1)
    if save_fig_path is not None:
        ax = cluster_score_df.loc[::, ["n_clusters", "silhouette_avg", "reverse_db_score", "ch_score", "synthetic_score"]].plot(title="cluster_score vs n_clusters",
                                                                                                                                x="n_clusters",
                                                                                                                                y=["silhouette_avg", "reverse_db_score", "ch_score", "synthetic_score"],
                                                                                                                                kind="line", grid=True, figsize=(10, 6))
        ax.set_xticks(n_clusters_list)
        fig = ax.get_figure()
        fig.savefig(save_fig_path)
        plt.show()
        plt.close()
    if various_n_clusters_model_info_df.shape[0] > 2:
        various_n_clusters_model_info_df.loc[::, "synthetic_score"] = cluster_score_df.loc[::, ["synthetic_score"]]
    various_n_clusters_model_info_df = various_n_clusters_model_info_df.set_index("n_clusters")
    DF_LOGGER.info("==================== various_n_clusters_model_info_df ====================")
    DF_LOGGER.info(various_n_clusters_model_info_df)


def filtered_small_n_samples_and_silhouette_min_cluster(clusters_info_df: pd.DataFrame, min_cluster_n_samples: int, min_cluster_silhouette: float):
    """
    Filtered small n_samples and silhouette min cluster
    """
    cluster_n_samples_mask = clusters_info_df.loc[::, 'cluster_n_samples'] > min_cluster_n_samples
    cluster_silhouette_min_mask = clusters_info_df.loc[::, 'cluster_silhouette_min'] > min_cluster_silhouette
    row_mask = cluster_n_samples_mask & cluster_silhouette_min_mask
    filtered_1_clusters_info_df = clusters_info_df.loc[row_mask, ::]
    filtered_cluster_labels = filtered_1_clusters_info_df.loc[::, "cluster_label"]
    dist_to_cluster_mask = filtered_1_clusters_info_df.columns.isin([f"dist_to_cluster_{label}" for label in filtered_cluster_labels])
    not_dist_to_cluster_mask = ~filtered_1_clusters_info_df.columns.str.contains("dist_to_cluster")
    col_mask = not_dist_to_cluster_mask | dist_to_cluster_mask
    final_filtered_clusters_info_df = clusters_info_df.loc[row_mask, col_mask]

    return final_filtered_clusters_info_df


def select_cluster_labels_with_max_dist(clusters_info_df: pd.DataFrame):
    """
    Select cluster labels with max distance
    """
    dist_to_cluster_mask = clusters_info_df.columns.str.contains("dist_to_cluster")
    clusters_dist_df = clusters_info_df.loc[::, dist_to_cluster_mask]
    assert issymmetric(clusters_dist_df.values, atol=1e-14), "clusters_dist_df is not symmetric"
    not_clusters_dist_df = clusters_info_df.loc[::, ~dist_to_cluster_mask]
    max_cluster_dist = clusters_dist_df.max().max()
    max_cluster_dist_mask = np.isclose(clusters_dist_df, max_cluster_dist).sum(axis=0).astype(bool)
    final_filtered_clusters_info_df = pd.concat([not_clusters_dist_df.loc[max_cluster_dist_mask, ::], clusters_dist_df.loc[max_cluster_dist_mask, max_cluster_dist_mask]], axis=1)
    selected_cluter_labels = final_filtered_clusters_info_df.loc[::, "cluster_label"].tolist()
    assert len(selected_cluter_labels) == 2, "len(selected_cluter_labels) is not equal to 2"
    assert final_filtered_clusters_info_df.shape[0] == 2, "final_filtered_clusters_info_df.shape[0] is not equal to 2"

    return selected_cluter_labels, max_cluster_dist, final_filtered_clusters_info_df


def pca_cluster(pca_input_data: pd.DataFrame, pca_kwargs: dict, cluster_kwargs: dict):
    reducted_data_df, pri_components = calc_pca(data=pca_input_data, n_samples=pca_kwargs["n_samples"],
                                                variance_thres=pca_kwargs["pca_explanation_variance_thres"], verbose=0)
    cluster_conditions = {"n_samples": pca_kwargs["n_samples"], "n_features": len(pri_components),
                          "linkage": cluster_kwargs["linkage"], "cluster_metric": cluster_kwargs["cluster_metric"]}
    hrchy_ret = calc_hrchy_cluster_given_n_clusters(n_clusters=cluster_kwargs["n_clusters"], data=reducted_data_df, cluster_conds=cluster_conditions)
    each_sample_cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values, db_score, ch_score, clusters_info_df = hrchy_ret
    filtered_1_clusters_info_df = filtered_small_n_samples_and_silhouette_min_cluster(clusters_info_df=clusters_info_df, min_cluster_n_samples=3, min_cluster_silhouette=0)
    selected_cluter_labels, max_cluster_dist, filtered_2_clusters_info_df = select_cluster_labels_with_max_dist(filtered_1_clusters_info_df)

    return selected_cluter_labels, max_cluster_dist, clusters_info_df, filtered_1_clusters_info_df, filtered_2_clusters_info_df, each_sample_cluster_labels
