import random
from itertools import combinations
from pathlib import Path
from pprint import pformat

import pandas as pd

from .assorted_utils import load_data_cfg
from .cluster_utils import (convert_pairs_data_to_proximity_mat,
                            filter_proximity_mat, pca_cluster)
from .log_utils import Log

LOGGER = Log().init_logger(logger_name=__name__)
DF_LOGGER = Log().init_logger(logger_name="df_logger")


def gen_random_items(all_items: list, ret_items_len: int = 100, verbose: int = 0, rand_seed: int = None):
    """
    Randon pick items for training # Not always necessary to operate
    """
    data_cfg = load_data_cfg()
    default_seed = data_cfg["RANDOM_SEEDS"]["DEFAUTL_SEED"]
    random.seed(rand_seed if rand_seed is not None else default_seed)
    ret_items = sorted(random.sample(all_items, ret_items_len))

    LOGGER.info(f"random seed: {rand_seed if rand_seed is not None else default_seed}")
    if verbose == 1:
        ori_logger_level = LOGGER.getEffectiveLevel()
        LOGGER.setLevel(10)
        LOGGER.debug(f"len(ret_items):{len(ret_items)}")
        LOGGER.debug(f"ret_items:\n{pformat(ret_items, width=500, compact=True)}")
        LOGGER.setLevel(ori_logger_level)

    return ret_items


def gen_corr_prop_filtered_items(item_pairs_ser: pd.Series, corr_prop_cond: str, item_names: tuple, fill_diag_val: int, ret_items_len: int, cliques_dir: Path, can_check_filtering_proc: bool):
    corr_prop_proximity_df = convert_pairs_data_to_proximity_mat(item_pairs_ser=item_pairs_ser, item_names=item_names, fill_diag_val=fill_diag_val)
    corr_prop_mask_settings = {"positive_corr_prop": (corr_prop_proximity_df > 0),
                               "negative_corr_prop": (corr_prop_proximity_df < 0)}
    corr_prop_mask = corr_prop_mask_settings[corr_prop_cond]
    corr_prop_filtered_proximity_df, _ = filter_proximity_mat(proximity_mat=corr_prop_proximity_df.copy(), filter_mask=corr_prop_mask, tmp_clique_dir=cliques_dir)
    corr_prop_filtered_items = corr_prop_filtered_proximity_df.columns.tolist()
    if len(corr_prop_filtered_items) > ret_items_len:
        ret_items = gen_random_items(corr_prop_filtered_items, ret_items_len, verbose=0)
    else:
        ret_items = corr_prop_filtered_items
    if can_check_filtering_proc:
        ori_logger_level = LOGGER.getEffectiveLevel()
        LOGGER.setLevel(10)
        LOGGER.debug("####################### check filtering precessing #######################")
        DF_LOGGER.debug("####################### item_pairs_ser #######################")
        DF_LOGGER.debug(item_pairs_ser)
        DF_LOGGER.debug("####################### corr_prop_proximity_df #######################")
        DF_LOGGER.debug(corr_prop_proximity_df)
        DF_LOGGER.debug("####################### corr_prop_filtered_proximity_df #######################")
        DF_LOGGER.debug(corr_prop_filtered_proximity_df)
        ret_item_pairs = [" & ".join(pair) for pair in combinations(ret_items, 2)]
        ret_item_pairs_mask = item_pairs_ser.index.isin(ret_item_pairs)
        DF_LOGGER.debug("####################### item_pairs_ser[filtered_item_pairs_mask] #######################")
        DF_LOGGER.debug(item_pairs_ser[ret_item_pairs_mask])
        LOGGER.setLevel(ori_logger_level)
    return ret_items


def gen_pca_cluster_filtered_items_each_cluster(pca_input_data: pd.DataFrame, pca_kwargs: dict, cluster_kwargs: dict):
    pca_cluster_ret = pca_cluster(pca_input_data=pca_input_data, pca_kwargs=pca_kwargs, cluster_kwargs=cluster_kwargs)
    selected_cluter_labels, max_cluster_dist, filtered_1_clusters_info_df, filtered_2_clusters_info_df, each_sample_cluster_labels = pca_cluster_ret
    ret_items_each_cluster = dict()
    pca_input_data_samples = pca_input_data.index
    all_input_items = pca_input_data_samples
    for cluster_label in selected_cluter_labels:
        ret_items_each_cluster.update({f"cluster_label_{cluster_label}": all_input_items[each_sample_cluster_labels == cluster_label].tolist()})

    LOGGER.info("================================= Info of pca_cluster_filtered_items_each_cluster  =================================")
    LOGGER.info(f"max_cluster_dist: {max_cluster_dist}")
    LOGGER.info(f"selected_cluter_labels: {selected_cluter_labels}")
    DF_LOGGER.info("========== filtered_1_clusters_info_df ==========")
    DF_LOGGER.info(filtered_1_clusters_info_df)
    DF_LOGGER.info("========== filtered_2_clusters_info_df ==========")
    DF_LOGGER.info(filtered_2_clusters_info_df)

    return ret_items_each_cluster


def gen_pca_cluster_filtered_pairs_each_cluster(pca_input_data: pd.DataFrame, pca_kwargs: dict, cluster_kwargs: dict):
    pca_cluster_ret = pca_cluster(pca_input_data=pca_input_data, pca_kwargs=pca_kwargs, cluster_kwargs=cluster_kwargs)
    selected_cluter_labels, max_cluster_dist, filtered_1_clusters_info_df, filtered_2_clusters_info_df, each_sample_cluster_labels = pca_cluster_ret
    ret_pairs_each_cluster = {}
    ret_pairs_idx_each_cluster = {}
    pca_input_data_samples = pca_input_data.index
    all_input_items = pca_input_data_samples
    comparison_df = pd.DataFrame({"pca_input_data_samples": pca_input_data_samples})
    for cluster_label in selected_cluter_labels:
        cl_key = f"cluster_label_{cluster_label}"
        ret_pairs_each_cluster.update({cl_key: all_input_items[each_sample_cluster_labels == cluster_label].tolist()})
        comparison_df[f"{cl_key}_pairs"] = comparison_df["pca_input_data_samples"].where(comparison_df["pca_input_data_samples"].isin(ret_pairs_each_cluster[cl_key]))
        ret_pairs_idx_each_cluster.update({cl_key: comparison_df[f"{cl_key}_pairs"].dropna().index.tolist()})
        assert len(ret_pairs_each_cluster) == len(ret_pairs_idx_each_cluster), "len(ret_pairs_each_cluster) != len(ret_pairs_idx_each_cluster)"
        assert len(ret_pairs_each_cluster[cl_key]) == len(ret_pairs_idx_each_cluster[cl_key]), f"len(ret_pairs_each_cluster[{cl_key}]) != len(ret_pairs_idx_each_cluster[{cl_key}])"
    all_ret_pairs = list(ret_pairs_each_cluster.values())
    ret_items = sorted(set([item for pair in all_ret_pairs for item in pair]))
    display_comparison_df = pd.concat([comparison_df.head(5), comparison_df.dropna(axis=0, thresh=2), comparison_df.tail(5)])

    LOGGER.info("================================= Info of pca_cluster_filtered_items_each_cluster  =================================")
    LOGGER.info(f"max_cluster_dist: {max_cluster_dist}")
    LOGGER.info(f"selected_cluter_labels: {selected_cluter_labels}")
    DF_LOGGER.info("========== filtered_1_clusters_info_df ==========")
    DF_LOGGER.info(filtered_1_clusters_info_df)
    DF_LOGGER.info("========== filtered_2_clusters_info_df ==========")
    DF_LOGGER.info(filtered_2_clusters_info_df)
    DF_LOGGER.info("========== display_comparison_df ==========")
    DF_LOGGER.info(display_comparison_df)

    return ret_pairs_each_cluster, ret_pairs_idx_each_cluster, ret_items
