import logging
from itertools import combinations
from pathlib import Path

import dynamic_yaml
import numpy as np
import pandas as pd
import yaml

from .etl_utils import calc_corr_ser_property
from .log_utils import Log

LOGGER = Log().init_logger(logger_name=__name__)
DF_LOGGER = Log().init_logger(logger_name="df_logger")

def split_and_norm_data(model_input_df: pd.DataFrame, batch_size: int, target_df: pd.DataFrame = None):
    """
    split dataset to train, validation, test
    normalize these dataset
    """
    num_pairs, all_timesteps = model_input_df.shape
    # Split to training, validation, and test sets
    model_input_mat = model_input_df.values
    for val_test_pct in np.linspace(0.1, 0.3, 21):
        if int(all_timesteps*val_test_pct) > 2*batch_size:
            train_pct = 1 - val_test_pct
            val_pct = train_pct + (val_test_pct / 2)
            break
    train_dataset = {"model_input": model_input_mat[::, :int(all_timesteps*train_pct)]}
    val_dataset = {"model_input": model_input_mat[::, int(all_timesteps*train_pct):int(all_timesteps*val_pct)]}
    test_dataset = {"model_input": model_input_mat[::, int(all_timesteps*val_pct):]}
    if target_df is not None:
        assert model_input_df.shape == target_df.shape, "Check the whether the shape of model_input_df and target_df are the same."
        target_mat = target_df.values
        train_dataset["target"] = target_mat[::, :int(all_timesteps*train_pct)]
        val_dataset["target"] = target_mat[::, int(all_timesteps*train_pct):int(all_timesteps*val_pct)]
        test_dataset["target"] = target_mat[::, int(all_timesteps*val_pct):]
    else:
        train_dataset["target"] = train_dataset["model_input"]
        val_dataset["target"] = val_dataset["model_input"]
        test_dataset["target"] = test_dataset["model_input"]

    LOGGER.info(f"split ratio: train:{train_pct}, val:{val_pct-train_pct}, test {1-val_pct}")
    LOGGER.info("="*80)

    return train_dataset, val_dataset, test_dataset


def find_abs_max_cross_corr(x):
    """Finds the index of absolute-maximum cross correlation of a signal with itself, then return the correspond cross correlation.

    Args:
      x: The signal.

    Returns:
      The sign*(abs_maximum cross correlation) of the signal with itself.
    """

    cross_correlation = np.correlate(x, x, mode='full')
    lag = np.argmax(np.absolute(cross_correlation))
    return cross_correlation[lag]


def find_cross_items_pairs(items_1_data_implement: str, items_2_data_implement: str, integrate_two_items: list, integrate_two_items_corr_df: pd.DataFrame) -> tuple[list, list, pd.DataFrame]:
    """Finds the cross items pairs."""
    data_cfg = load_data_cfg()
    items_1 = data_cfg["DATASETS"][items_1_data_implement]['TRAIN_SET']
    items_2 = data_cfg["DATASETS"][items_2_data_implement]['TRAIN_SET']
    assert integrate_two_items == sorted(items_1+items_2), f"integrate_two_items != sorted(items_1+items_2), integrate_two_items: {integrate_two_items}, sorted(items_1+items_2): {sorted(items_1+items_2)}"


    items_1_pairs = [f"{pair[0]} & {pair[1]}" for pair in combinations(items_1, r=2)]
    items_2_pairs = [f"{pair[0]} & {pair[1]}" for pair in combinations(items_2, r=2)]
    merge_two_pairs = items_1_pairs+items_2_pairs
    integrate_two_items_pairs = [f"{pair[0]} & {pair[1]}" for pair in combinations(sorted(items_1+items_2), r=2)]
    comparison_df = pd.DataFrame({"corr_df_pairs": integrate_two_items_corr_df.index.tolist(), "integrate_two_items_pairs": integrate_two_items_pairs})
    assert all(comparison_df["corr_df_pairs"] == comparison_df["integrate_two_items_pairs"])
    comparison_df["items_1_pairs"] = comparison_df["integrate_two_items_pairs"].where(cond=comparison_df["integrate_two_items_pairs"].isin(items_1_pairs))
    comparison_df["items_2_pairs"] = comparison_df["integrate_two_items_pairs"].where(cond=comparison_df["integrate_two_items_pairs"].isin(items_2_pairs))
    comparison_df["merge_two_pairs"] = comparison_df["integrate_two_items_pairs"].where(cond=comparison_df["integrate_two_items_pairs"].isin(merge_two_pairs))
    comparison_df["cross_items_pairs"] =  comparison_df["integrate_two_items_pairs"].where(cond=~comparison_df["integrate_two_items_pairs"].isin(merge_two_pairs))
    cross_items_pairs = comparison_df["cross_items_pairs"].dropna().tolist()
    cross_items_pairs_idx = comparison_df.index.where(cond=comparison_df["cross_items_pairs"].notnull()).dropna().astype(int).tolist()
    assert len(cross_items_pairs) == len(cross_items_pairs_idx), "len(cross_items_pairs) != len(cross_items_pairs_idx)"
    assert len(cross_items_pairs)+len(merge_two_pairs) == len(integrate_two_items_pairs), "len(cross_items_pairs)+len(merge_two_pairs) != len(integrate_two_items_pairs)"

    LOGGER.info(f"integrate_two_items: {integrate_two_items}")
    LOGGER.info(f"items_1: {items_1}")
    LOGGER.info(f"items_2: {items_2}")
    LOGGER.info(f"len(items_1): {len(items_1)}, len(items_1_pairs): {len(items_1_pairs)}, len(items_2): {len(items_2)}, len(items_2_pairs): {len(items_2_pairs)}")
    LOGGER.info(f"len(merge_two_pairs): {len(merge_two_pairs)}, len(integrate_two_items_pairs): {len(integrate_two_items_pairs)}, len(cross_items_pairs): {len(cross_items_pairs)}")
    DF_LOGGER.info("==============================Info of comparison_df ==============================")
    DF_LOGGER.info(comparison_df)

    return cross_items_pairs, cross_items_pairs_idx, comparison_df


def convert_str_bins_list(str_bins: str) -> list:
    """Converts a string of bins to a list of bins.

    Args:
      str_bins: A string of bins.

    Returns:
      A list of bins.
    """

    bins_list = []
    for str_bin in str_bins.replace("bins_", "").split("_"):
        if "-" in str_bin:
            new_str_bin = str_bin[:2]+"."+str_bin[2:]
            bins_list.append(float(new_str_bin))
        else:
            new_str_bin = str_bin[:1]+"."+str_bin[1:]
            bins_list.append(float(new_str_bin))

    return bins_list


def load_data_cfg():
    """
    Load data config file.
    """
    this_file_dir = Path(__file__).parent
    data_config_path = this_file_dir/"../config/data_config.yaml"
    with open(data_config_path) as f:
        data = dynamic_yaml.load(f)
        data_cfg = yaml.full_load(dynamic_yaml.dump(data))
    return data_cfg


def get_certain_level_dict_items(nested_dict: dict, lvl: int):
    """
    Get the items of certain level of nested dict
    """
    if lvl == 1:
        return nested_dict.items()
    elif all(isinstance(sub_dict, dict) for sub_dict in nested_dict.values()):
        return [item for sub_dict in nested_dict.values() for item in get_certain_level_dict_items(nested_dict=sub_dict, lvl=lvl-1)]
    else:
        LOGGER.error("input_dict is not nested_dict")


def get_certain_level_dict_values_given_key(nested_dict: dict, lvl: int, key: str):
    """
    Get the values of certain level of nested dict given key
    """
    lvl_items = get_certain_level_dict_items(nested_dict=nested_dict, lvl=lvl)
    ret_values = []
    for item in lvl_items:
        if item[0] == key and isinstance(item[1], list):
            ret_values.extend(item[1])
        elif item[0] == key and not isinstance(item[1], list):
            LOGGER.error(f"item[1] is not list, item[0]: {item[0]}, key: {key}, item[1]: {item[1]}")

    return ret_values


def load_multiple_data(data_implement: str, retrieve_items_setting: str, corr_type: str, target_df_bins: str, w_l: int, s_l: int, corr_ser_clac_method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads multiple data.

    Args:
      data_implement: The data implement.
      retrieve_items_setting: The retrieve items setting.
      corr_type: The correlation type.
      w_l: The window length of correlation series.
      s_l: The stride length of correlation series.

    Returns:
      The dataset dataframe, correlation dataframe, target dataframe, and correlation series property dataframe.
    """
    data_cfg = load_data_cfg()
    all_set = data_cfg["DATASETS"][data_implement]['ALL_SET']  # all items
    train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
    items_implement = train_set if retrieve_items_setting == "-train_train" else all_set
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + retrieve_items_setting
    _, corr_dir, target_dir, corr_property_dir, _, _ = load_dirs(data_implement=data_implement,
                                                                 retrieve_items_setting=retrieve_items_setting,
                                                                 corr_type=corr_type, target_df_bins=target_df_bins,
                                                                 w_l=w_l, s_l=s_l,
                                                                 corr_ser_clac_method=corr_ser_clac_method)
    corr_df_path = corr_dir/f"corr_s{s_l}_w{w_l}.csv"
    target_df_path = target_dir/f"corr_s{s_l}_w{w_l}.csv"
    corr_property_df_path = corr_property_dir/"corr_series_property.csv"
    dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
    dataset_df = dataset_df.set_index('Date')
    dataset_df = dataset_df.loc[::, items_implement]
    corr_df = pd.read_csv(corr_df_path, index_col=["item_pair"])
    target_df = pd.read_csv(target_df_path, index_col=["item_pair"])
    corr_property_df = calc_corr_ser_property(corr_dataset=corr_df, corr_property_df_path=corr_property_df_path)
    assert all(corr_df.index == target_df.index) and all(corr_df.index == corr_property_df.index), "Check the whether the index of corr_df, target_df, and corr_property_df are the same."

    LOGGER.info("=========================================== Info of Datasets ===========================================")
    LOGGER.info(f"len(items_implement): {len(items_implement)} and len(all_set): {len(all_set if all_set else [])} and len(train_set): {len(train_set if train_set else [])}")
    LOGGER.info(f"output_file_name: {output_file_name}, corr_s{s_l}_w{w_l} and corr_ser_clac_method:{corr_ser_clac_method}")
    LOGGER.info(f"dataset_df.shape:{dataset_df.shape}, corr_df.shape:{corr_df.shape}, target_df.shape:{target_df.shape}")
    LOGGER.info(f"corr_property_df.shape: {corr_property_df.shape}")
    LOGGER.info(f"Min of corr_ser_mean:{corr_property_df.loc[::,'corr_ser_mean'].min()}, Max of corr_ser_mean:{corr_property_df.loc[::,'corr_ser_mean'].max()}")

    return dataset_df, corr_df, target_df, corr_property_df


def load_dirs(data_implement: str, retrieve_items_setting: str, corr_type: str, target_df_bins: str, w_l: int, s_l: int, corr_ser_clac_method: str) -> tuple[Path, Path, Path, Path, Path]:
    """
    Load directories of correlation data.
    """

    data_cfg = load_data_cfg()
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + retrieve_items_setting
    pipeline_corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{corr_type}"
    corr_dir = pipeline_corr_data_dir/"corr_data"
    target_dir = pipeline_corr_data_dir/f"custom_discretize_corr_data/{target_df_bins}"
    corr_property_dir = pipeline_corr_data_dir/f"corr_property/corr_s{s_l}_w{w_l}/{corr_ser_clac_method}"
    cliques_dir = pipeline_corr_data_dir/f"cliques/corr_s{s_l}_w{w_l}/{corr_ser_clac_method}"
    clusters_dir = pipeline_corr_data_dir/f"clusters/corr_s{s_l}_w{w_l}"

    return pipeline_corr_data_dir, corr_dir, target_dir, corr_property_dir, cliques_dir, clusters_dir
