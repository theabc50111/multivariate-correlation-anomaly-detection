import random
from itertools import combinations
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from .assorted_utils import (get_certain_level_dict_values_given_key,
                             load_data_cfg)
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

    if verbose == 1:
        ori_logger_level = LOGGER.getEffectiveLevel()
        LOGGER.setLevel(10)
        LOGGER.debug(f"random seed: {rand_seed if rand_seed is not None else default_seed}")
        LOGGER.debug(f"len(ret_items):{len(ret_items)}")
        LOGGER.debug(f"ret_items:\n{pformat(ret_items, width=500, compact=True)}")
        LOGGER.setLevel(ori_logger_level)

    return ret_items


def gen_corr_prop_filtered_items(item_pairs_ser: pd.Series, corr_prop_cond: str, item_names: tuple, fill_diag_val: int, ret_items_len: int, cliques_dir: Path, can_check_filtering_proc: bool):
    corr_prop_proximity_df = convert_pairs_data_to_proximity_mat(item_pairs_ser=item_pairs_ser, item_names=item_names, fill_diag_val=fill_diag_val)
    corr_prop_mask_settings = {"strong_positive_corr_prop": (corr_prop_proximity_df > 0.7),
                               "moderate_positive_corr_prop": (corr_prop_proximity_df < 0.7) & (corr_prop_proximity_df > 0.3),
                               "above_moderate_positive_corr_prop": (corr_prop_proximity_df > 0.3),
                               "below_moderate_positive_corr_prop": (corr_prop_proximity_df < 0.3),
                               "positive_corr_prop": (corr_prop_proximity_df > 0),
                               "negative_corr_prop": (corr_prop_proximity_df < 0)}
    corr_prop_mask = corr_prop_mask_settings[corr_prop_cond]
    corr_prop_filtered_proximity_df, _ = filter_proximity_mat(proximity_mat=corr_prop_proximity_df.copy(), filter_mask=corr_prop_mask, tmp_clique_dir=cliques_dir)
    corr_prop_filtered_items = corr_prop_filtered_proximity_df.columns.tolist()
    if len(corr_prop_filtered_items) > ret_items_len:
        ret_items = gen_random_items(all_items=corr_prop_filtered_items, ret_items_len=ret_items_len, verbose=0)
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


def gen_pca_cluster_filtered_samples_two_max_dist_clusters(pca_input_data: pd.DataFrame, pca_kwargs: dict, cluster_kwargs: dict, filter_on: str, verbose: int = 0):
    """
    Generate filtered samples by pca_cluster(), and return samples in two clusters with max distance
    """
    pca_cluster_ret = pca_cluster(pca_input_data=pca_input_data, pca_kwargs=pca_kwargs, cluster_kwargs=cluster_kwargs)
    selected_cluter_labels, max_cluster_dist, clusters_info_df, filtered_1_clusters_info_df, filtered_2_clusters_info_df, each_sample_cluster_labels = pca_cluster_ret
    ret_samples_two_max_dist_cluster = {}
    pca_input_data_samples = pca_input_data.index
    comparison_df = pd.DataFrame({"pca_input_data_samples_idx": list(range(len(pca_input_data_samples))), "pca_input_data_samples": pca_input_data_samples})
    for cluster_label in selected_cluter_labels:
        cl_key = f"cluster_label_{cluster_label}"
        selected_cluster_samples = pca_input_data_samples[each_sample_cluster_labels == cluster_label].tolist()
        selected_cluster_samples_idx = (each_sample_cluster_labels == cluster_label).nonzero()[0].tolist()
        ret_samples_two_max_dist_cluster.update({cl_key: {"samples": selected_cluster_samples,
                                                          "samples_idx": selected_cluster_samples_idx}})
        comparison_df[f"{cl_key}_items"] = comparison_df["pca_input_data_samples"].where(comparison_df["pca_input_data_samples"].isin(ret_samples_two_max_dist_cluster[cl_key]["samples"]))
    integrated_samples_each_cluster = sorted(get_certain_level_dict_values_given_key(nested_dict=ret_samples_two_max_dist_cluster, lvl=2, key="samples"))
    integrated_samples_idx_each_cluster = sorted(get_certain_level_dict_values_given_key(nested_dict=ret_samples_two_max_dist_cluster, lvl=2, key="samples_idx"))
    display_comparison_df = pd.concat([comparison_df.head(5), comparison_df.dropna(axis=0, thresh=3), comparison_df.tail(5)])

    LOGGER.info("================================= Info of pca_cluster_filtered_samples_two_max_dist_clusters =================================")
    LOGGER.info(f"max_cluster_dist: {max_cluster_dist}")
    LOGGER.info(f"selected_cluter_labels: {selected_cluter_labels}")
    DF_LOGGER.info("========== filtered_1_clusters_info_df ==========")
    DF_LOGGER.info(filtered_1_clusters_info_df)
    DF_LOGGER.info("========== filtered_2_clusters_info_df ==========")
    DF_LOGGER.info(filtered_2_clusters_info_df)
    DF_LOGGER.info("========== display_comparison_df ==========")
    DF_LOGGER.info(display_comparison_df)
    if verbose == 1:
        DF_LOGGER.debug("========== clusters_info_df ==========")
        DF_LOGGER.debug(clusters_info_df)

    if filter_on == "items":
        assert pca_input_data_samples[0].count(" & ") == 0, f"pca_input_data_samples[0] should not contains ' & ', pca_input_data_samples[0]: {pca_input_data_samples[0]}"
        integrated_items = {"items": integrated_samples_each_cluster, "items_idx": integrated_samples_idx_each_cluster}

        return ret_samples_two_max_dist_cluster, integrated_items
    elif filter_on == "pairs":
        assert pca_input_data_samples[0].count(" & ") == 1, f"pca_input_data_samples[0] should contains ' & ', pca_input_data_samples[0]: {pca_input_data_samples[0]}"
        integrated_items_each_cluster = sorted(set([item for pair in integrated_samples_each_cluster for item in pair.split(" & ")]))
        integrated_pairs = {"pairs": integrated_samples_each_cluster, "pairs_idx": integrated_samples_idx_each_cluster, "items": integrated_items_each_cluster}

        return ret_samples_two_max_dist_cluster, integrated_pairs


def gen_pca_cluster_filtered_samples_each_cluster(pca_input_data: pd.DataFrame, num_selected_clusters: int, num_samples_each_cluster: int, pca_kwargs: dict, cluster_kwargs: dict, filter_on: str):
    """
    Generate filtered items by pca_cluster(), and return items in each cluster
    """
    _, _, _, filtered_1_clusters_info_df, _, each_sample_cluster_labels = pca_cluster(pca_input_data=pca_input_data, pca_kwargs=pca_kwargs, cluster_kwargs=cluster_kwargs)
    filtered_cluster_labels = filtered_1_clusters_info_df.loc[::, 'cluster_label'].tolist()
    assert len(filtered_cluster_labels) >= num_selected_clusters, f"len(filtered_cluster_labels): {len(filtered_cluster_labels)} should >= num_selected_clusters: {num_selected_clusters}"
    selected_cluter_labels = gen_random_items(all_items=filtered_cluster_labels, ret_items_len=num_selected_clusters, verbose=0)
    ret_samples_each_cluster = {}
    pca_input_data_samples = pca_input_data.index
    selected_clusters_samples_df = pd.DataFrame()
    comparison_df = pd.DataFrame({"pca_input_data_samples_idx": list(range(len(pca_input_data_samples))), "pca_input_data_samples": pca_input_data_samples})
    for cluster_label in selected_cluter_labels:
        cl_key = f"cluster_label_{cluster_label}"
        selected_cluster_samples = pca_input_data_samples[each_sample_cluster_labels == cluster_label].tolist()
        selected_samples = gen_random_items(all_items=selected_cluster_samples, ret_items_len=num_samples_each_cluster, verbose=0)
        selected_samples_idx = pca_input_data_samples.isin(selected_samples).nonzero()[0].tolist()
        ret_samples_each_cluster.update({cl_key: {"samples": selected_samples,
                                                  "samples_idx": selected_samples_idx}})
        comparison_df[f"{cl_key}_samples"] = comparison_df["pca_input_data_samples"].where(comparison_df["pca_input_data_samples"].isin(ret_samples_each_cluster[cl_key]["samples"]))
        selected_clusters_samples_df = pd.concat([selected_clusters_samples_df, pd.DataFrame({f"{cl_key}_samples": selected_cluster_samples})], axis=1)
    integrated_samples_each_cluster = sorted(get_certain_level_dict_values_given_key(nested_dict=ret_samples_each_cluster, lvl=2, key="samples"))
    integrated_samples_idx_each_cluster = sorted(get_certain_level_dict_values_given_key(nested_dict=ret_samples_each_cluster, lvl=2, key="samples_idx"))
    display_comparison_df = comparison_df.dropna(axis=0, thresh=3)

    LOGGER.info("================================= Info of pca_cluster_filtered_samples_each_cluster  =================================")
    LOGGER.info(f"selected_cluter_labels: {selected_cluter_labels}")
    DF_LOGGER.info("========== filtered_1_clusters_info_df ==========")
    DF_LOGGER.info(filtered_1_clusters_info_df)
    DF_LOGGER.info("========== selected_clusters_samples_df ==========")
    DF_LOGGER.info(selected_clusters_samples_df)
    DF_LOGGER.info("========== display_comparison_df ==========")
    DF_LOGGER.info(display_comparison_df)

    if filter_on == "items":
        assert pca_input_data_samples[0].count(" & ") == 0, f"pca_input_data_samples[0] should not contains ' & ', pca_input_data_samples[0]: {pca_input_data_samples[0]}"
        integrated_items = {"items": integrated_samples_each_cluster, "items_idx": integrated_samples_idx_each_cluster}

        return ret_samples_each_cluster, integrated_items
    elif filter_on == "pairs":
        assert pca_input_data_samples[0].count(" & ") == 1, f"pca_input_data_samples[0] should contains ' & ', pca_input_data_samples[0]: {pca_input_data_samples[0]}"
        integrated_items_each_cluster = sorted(set([item for pair in integrated_samples_each_cluster for item in pair.split(" & ")]))
        integrated_pairs = {"pairs": integrated_samples_each_cluster, "pairs_idx": integrated_samples_idx_each_cluster, "items": integrated_items_each_cluster}

        return ret_samples_each_cluster, integrated_pairs


def gen_pca_cluster_samples_with_given_sample(pca_input_data: pd.DataFrame, given_sample: str, pca_kwargs: dict, cluster_kwargs: dict, filter_on: str):
    """
    Generate filtered items by pca_cluster(), and return items in each cluster
    """

    pca_input_data_samples = pca_input_data.index
    given_sample_mask = pca_input_data_samples == given_sample
    given_sample_idx = given_sample_mask.nonzero()[0][0]
    _, _, clusters_info_df, _, _, each_sample_cluster_labels = pca_cluster(pca_input_data=pca_input_data, pca_kwargs=pca_kwargs, cluster_kwargs=cluster_kwargs)
    selected_cluster_label = each_sample_cluster_labels[given_sample_idx]
    selected_samples = pca_input_data_samples[each_sample_cluster_labels == selected_cluster_label].tolist()
    selected_samples_idx = pca_input_data_samples.isin(selected_samples).nonzero()[0].tolist()
    comparison_df = pd.DataFrame({"pca_input_data_samples_idx": list(range(len(pca_input_data_samples))), "pca_input_data_samples": pca_input_data_samples})
    ret_cluster_samples = {f"cluster_label_{selected_cluster_label}": {"samples": selected_samples,
                                                                       "samples_idx": selected_samples_idx}}
    comparison_df[f"cluster_label_{selected_cluster_label}_samples"] = comparison_df["pca_input_data_samples"].where(comparison_df["pca_input_data_samples"].isin(ret_cluster_samples[f'cluster_label_{selected_cluster_label}']['samples']))
    selected_cluster_info_df = clusters_info_df.where(clusters_info_df["cluster_label"] == selected_cluster_label).dropna(axis=0, how="all")
    display_comparison_df = comparison_df.dropna(axis=0, thresh=3)
    if filter_on == "items":
        assert pca_input_data.index[0].count(" & ") == 0 and given_sample.count(" & ") == 0, f"pca_input_data.index and given_sample should not contains ' & ', given_sample: {given_sample}, pca_input_data.index[0]: {pca_input_data.index[0]}"
    elif filter_on == "pairs":
        assert pca_input_data.index[0].count(" & ") == 1 and given_sample.count(" & ") == 1, f"pca_input_data.index and given_sample should contains ' & ', given_sample: {given_sample}, pca_input_data.index[0]: {pca_input_data.index[0]}"
        ret_cluster_samples[f"cluster_label_{selected_cluster_label}"]["items"] = sorted(set([item for pair in selected_samples for item in pair.split(" & ")]))

    LOGGER.info(f"given_sample: {given_sample}, given_sample_idx: {given_sample_idx}, pca_input_data_samples[{given_sample_idx}]: {pca_input_data_samples[given_sample_idx]}, selected_cluster_label: {selected_cluster_label}")
    DF_LOGGER.info("========== selected_cluster_info_df ==========")
    DF_LOGGER.info(selected_cluster_info_df)
    DF_LOGGER.info("========== display_comparison_df ==========")
    DF_LOGGER.info(display_comparison_df)

    return ret_cluster_samples


def gen_pca_cluster_samples(pca_input_data: pd.DataFrame, pca_kwargs: dict, cluster_kwargs: dict):
    """
    Generate samples by pca_cluster(), and return samples of each cluster
    """
    _, _, clusters_info_df, _, _, each_sample_cluster_labels = pca_cluster(pca_input_data=pca_input_data, pca_kwargs=pca_kwargs, cluster_kwargs=cluster_kwargs)
    pca_input_data_samples = pca_input_data.index
    all_clusters_labels = clusters_info_df.loc[::, 'cluster_label'].tolist()
    all_clusters_samples_df = pd.DataFrame()
    ret_samples_each_cluster = {}
    comparison_df = pd.DataFrame({"pca_input_data_samples_idx": list(range(len(pca_input_data_samples))), "pca_input_data_samples": pca_input_data_samples})
    for cluster_label in all_clusters_labels:
        cl_key = f"cluster_label_{cluster_label}"
        cluster_samples = pca_input_data_samples[each_sample_cluster_labels == cluster_label].tolist()
        cluster_samples_idx = pca_input_data_samples.isin(cluster_samples).nonzero()[0].tolist()
        ret_samples_each_cluster.update({cl_key: {"samples": cluster_samples,
                                                  "samples_idx": cluster_samples_idx}})
        comparison_df[f"{cl_key}_samples"] = comparison_df["pca_input_data_samples"].where(comparison_df["pca_input_data_samples"].isin(ret_samples_each_cluster[cl_key]["samples"]))
        all_clusters_samples_df = pd.concat([all_clusters_samples_df, pd.DataFrame({f"{cl_key}_samples": cluster_samples})], axis=1)
    display_comparison_df = comparison_df.dropna(axis=0, thresh=3)

    LOGGER.info("================================= Info of pca_cluster_filtered_samples_each_cluster  =================================")
    DF_LOGGER.info("========== all_clusters_samples_df ==========")
    DF_LOGGER.info(all_clusters_samples_df)
    DF_LOGGER.info("========== display_comparison_df ==========")
    DF_LOGGER.info(display_comparison_df)

    return ret_samples_each_cluster
