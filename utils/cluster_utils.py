# Function: clustering utilities
import logging
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import uniform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)
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
        logger.debug(f"{i}th clique: {clique}")
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
            logger.warn(f"clique search time out: {now_t - train_start_t} seconds")
            break
    top_3_len_cliq_cnt_key = sorted(clique_counter, key=lambda x: int(x.split('_')[1]), reverse=True)[:3]
    top_3_len_cliq = [f"number of {clique_len}: num" for clique_len in top_3_len_cliq_cnt_key]
    top_3_freq_cliq_len = [f"number of {clique_len}: {num}" for clique_len, num in clique_counter.most_common(3)]
    logger.info(f"total cliques: {clique_counter.total()}, top 3 num_cliques by frequent of len: {top_3_freq_cliq_len} top 3 cliques by len: {top_3_len_cliq}")
    proximity_mat = proximity_mat.loc[max_clique, max_clique]
    np.fill_diagonal(proximity_mat.values, ori_diag_val)
    return proximity_mat, max_clique


def calc_silhouette_label_freq_std(estimator: sklearn.base.ClusterMixin, x: pd.DataFrame, silhouette_score_ratio: int = 0.1, silhouette_metric: str = "precomputed") -> float:
    estimator.fit(x)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(x.index)
    labels_symbol, label_freq = np.unique(cluster_labels, return_counts=True)
    if num_labels in (1, num_samples):
        return -1

    return silhouette_score_ratio * silhouette_score(x, cluster_labels, metric=silhouette_metric) + (1 - silhouette_score_ratio) * (1 / np.array(label_freq).std())


def hrchy_clustering_distance_threshold_rs(x: pd.DataFrame, data_mat_mode: str = "precomputed", verbose: int = 0):
    param_dict = {"n_clusters": [None], "affinity": [data_mat_mode],
                  "linkage": ["single", "complete", "average"],
                  "distance_threshold": uniform(loc=0.55, scale=0.6),
                  "compute_distances": True}
    cv = [(slice(None), slice(None))]
    hrchy_clustering_rs = RandomizedSearchCV(estimator=AgglomerativeClustering(), param_distributions=param_dict,
                                             n_iter=100000, scoring=calc_silhouette_label_freq_std, cv=cv, n_jobs=-1)
    hrchy_clustering_rs.fit(x)

    if verbose == 1:
        print(f"hrchy_clustering_rs.best_estimator_: {hrchy_clustering_rs.best_estimator_}")
        print(f"hrchy_clustering_rs.best_params_: {hrchy_clustering_rs.best_params_}")
        print(f"hrchy_clustering_rs.best_score_: {hrchy_clustering_rs.best_score_}")
        print(f"hrchy_clustering_rs.best_estimator_.n_leaves_: {hrchy_clustering_rs.best_estimator_.n_leaves_}")
        print(f"hrchy_clustering_rs.best_estimator_.n_clusters_: {hrchy_clustering_rs.best_estimator_.n_clusters_}")
        print(f"np.unique(hrchy_clustering_rs.best_estimator_.labels_): {np.unique(hrchy_clustering_rs.best_estimator_.labels_, return_counts=True)}")
        print(f"hrchy_clustering_rs.best_estimator_.labels_: {hrchy_clustering_rs.best_estimator_.labels_}")
        print(f"hrchy_clustering_rs.n_features_in_: {hrchy_clustering_rs.n_features_in_}")
        print(f"hrchy_clustering_rs.feature_names_in_: {hrchy_clustering_rs.feature_names_in_}")
        print("-"*50)

    return hrchy_clustering_rs.best_estimator_


def hrchy_clustering_n_cluster_gs(x: pd.DataFrame, data_mat_mode: str = "precomputed", verbose: int = 0):
    param_dict = {"n_clusters": range(2, 20), "affinity": [data_mat_mode],
                  "linkage": ["single", "complete", "average"],
                  "compute_distances": True}
    cv = [(slice(None), slice(None))]
    hrchy_clustering_gs = GridSearchCV(estimator=AgglomerativeClustering(), param_grid=param_dict,
                                       scoring=calc_silhouette_label_freq_std, cv=cv, n_jobs=-1)
    hrchy_clustering_gs.fit(x)

    if verbose == 1:
        print(f"hrchy_clustering_gs.best_estimator_: {hrchy_clustering_gs.best_estimator_}")
        print(f"hrchy_clustering_gs.best_params_: {hrchy_clustering_gs.best_params_}")
        print(f"hrchy_clustering_gs.best_score_: {hrchy_clustering_gs.best_score_}")
        print(f"hrchy_clustering_gs.best_estimator_.n_leaves_: {hrchy_clustering_gs.best_estimator_.n_leaves_}")
        print(f"hrchy_clustering_gs.best_estimator_.n_clusters_: {hrchy_clustering_gs.best_estimator_.n_clusters_}")
        print(f"np.unique(hrchy_clustering_gs.best_estimator_.labels_): {np.unique(hrchy_clustering_gs.best_estimator_.labels_, return_counts=True)}")
        print(f"hrchy_clustering_gs.best_estimator_.labels_: {hrchy_clustering_gs.best_estimator_.labels_}")
        print(f"hrchy_clustering_gs.n_features_in_: {hrchy_clustering_gs.n_features_in_}")
        print(f"hrchy_clustering_gs.feature_names_in_: {hrchy_clustering_gs.feature_names_in_}")
        print("-"*50)

    return hrchy_clustering_gs.best_estimator_


def obs_hrchy_cluster_instances(x: pd.DataFrame, data_mat_mode: str = "precomputed", verbose: int = 1):
    for n in range(2, 20):
        hrchy_cluster = AgglomerativeClustering(n_clusters=n, linkage="complete", affinity=data_mat_mode, compute_distances=True)
        hrchy_cluster.fit(x)

        if verbose == 1:
            print(f"hrchy_cluste.n_clusters_: {hrchy_cluster.n_clusters_}")
            print(f"hrchy_cluste.labels and whose number of instances: {np.unique(hrchy_cluster.labels_, return_counts=True)}")
            # print(f"(ticker, cluster label): {list(zip(X.index, hrchy_cluster.labels_))}")
            # print(f"The estimated number of connected components:{hrchy_cluster.n_connected_components_}")
            # print(f"hrchy_cluster.n_leaves_: {hrchy_cluster.n_leaves_}")
            # print(f"hrchy_cluster.n_features_in_: {hrchy_cluster.n_features_in_}")
            print("-"*50)


def hrchy_cluster_fixed_n_cluster(x: pd.DataFrame, n: int, data_mat_mode: str = "precomputed", verbose: int = 1):
    hrchy_cluster = AgglomerativeClustering(n_clusters=n, linkage="complete", affinity=data_mat_mode, compute_distances=True)
    hrchy_cluster.fit(x)

    if verbose == 1:
        print(f"hrchy_cluste.n_clusters_: {hrchy_cluster.n_clusters_}")
        print(f"hrchy_cluste.labels and whose instances: {np.unique(hrchy_cluster.labels_, return_counts=True)}")
        print(f"hrchy_cluster.n_leaves_: {hrchy_cluster.n_leaves_}")
        print(f"hrchy_cluster.n_features_in_: {hrchy_cluster.n_features_in_}")
        print("-"*50)

    return hrchy_cluster


