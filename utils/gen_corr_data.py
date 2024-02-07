import argparse
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from utils.assorted_utils import load_data_cfg
from utils.log_utils import Log

DATA_CFG = load_data_cfg()
LOGGER = Log().init_logger(logger_name=__name__)
DF_LOGGER = Log().init_logger(logger_name="df_logger")
warnings.simplefilter("ignore")

def set_corr_data(data_implement, data_cfg: dict, data_gen_cfg: dict, corr_data_dir: Path, train_items_setting: str = "train_train", save_corr_data: bool = False):
    """
    # Data implement & output setting & testset setting
          data_implement: data implement setting  # watch options by operate: print(data_cfg["DATASETS"].keys())
          data_cfg: dict of pre-processed-data info, which is from 「config/data_config.yaml」
          data_gen_cfg: dict data generation configuration
          train_items_setting: train set setting  # train_train|train_all
          save_corr_data: setting of output files
    """

    # data loading & implement setting
    dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'], parse_dates=['Date'], index_col=['Date']).sort_index(axis=1).sort_index(axis=0)
    all_set = data_cfg["DATASETS"][data_implement]['ALL_SET']
    train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
    items_implement = train_set if train_items_setting == "train_train" else all_set
    dataset_df = dataset_df.loc[::, items_implement]
    LOGGER.info(f"===== len(all_set): {len(all_set)}, len(train_set): {len(train_set)}, len(items_implement): {len(items_implement)} =====")
    DF_LOGGER.info("========== overview dataset_df ==========")
    DF_LOGGER.info(dataset_df)

    # Load or Create Correlation Data
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}.csv"
    if corr_df_path.exists():
        corr_dataset = pd.read_csv(corr_df_path, index_col=["item_pair"])
        corr_dataset.columns = pd.DatetimeIndex(corr_dataset.columns)
    else:
        tmp_df = dataset_df.rolling(window=w_l).corr().dropna().unstack(level=0).stack(level=0).iloc[::, ::s_l]
        tmp_df["item_pair"] = tmp_df.index.get_level_values(0) + " & " + tmp_df.index.get_level_values(1)
        tmp_df = tmp_df.set_index(["item_pair"], drop=True)
        tmp_df.columns = pd.DatetimeIndex(tmp_df.columns)
        item_pairs = [f"{pair[0]} & {pair[1]}" for pair in combinations(sorted(dataset_df.columns), r=2)]
        item_pairs_mask = tmp_df.index.isin(item_pairs)
        corr_dataset = tmp_df.iloc[item_pairs_mask, ::]
    if save_corr_data:
        corr_dataset.to_csv(corr_df_path)

    DF_LOGGER.info("========== overview corr_dataset ==========")
    DF_LOGGER.info(corr_dataset.head())
    return corr_dataset


def set_certain_pairs_corr_data(data_implement, data_cfg: dict, data_gen_cfg: dict, corr_data_dir: Path, save_corr_data: bool = False):
    # data loading & implement setting
    ori_corr_df_dir = Path(data_cfg["DATASETS"][data_implement]['FILE_PATH']).parent
    ori_corr_df_path = ori_corr_df_dir/f"corr_s{data_gen_cfg['CORR_STRIDE']}_w{data_gen_cfg['CORR_WINDOW']}.csv"
    ori_corr_df = pd.read_csv(ori_corr_df_path, index_col=['item_pair']).sort_index(axis=1).sort_index(axis=0)
    pairs_implement = data_cfg["DATASETS"][data_implement]['TRAIN_PAIRS_SET']
    LOGGER.info(f"===== ori_corr_df.shape: {ori_corr_df.shape}, len(pairs_implement): {len(pairs_implement)} =====")
    DF_LOGGER.info("========== overview ori_corr_df ==========")
    DF_LOGGER.info(ori_corr_df)

    # Load or Create Correlation Data
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}.csv"
    if corr_df_path.exists():
        corr_dataset = pd.read_csv(corr_df_path, index_col=["item_pair"])
        corr_dataset.columns = pd.DatetimeIndex(corr_dataset.columns)
    else:
        corr_dataset = ori_corr_df.loc[pairs_implement, ::]
        corr_dataset.columns = pd.DatetimeIndex(corr_dataset.columns)
    assert corr_dataset.shape[0] == len(pairs_implement), f"corr_dataset.shape[0]:{corr_dataset.shape[0]} != len(pairs_implement):{len(pairs_implement)}"
    if save_corr_data:
        corr_dataset.to_csv(corr_df_path)

    DF_LOGGER.info("========== overview corr_dataset ==========")
    DF_LOGGER.info(corr_dataset.head())
    return corr_dataset


def gen_custom_discretize_corr(src_dir: Path, data_gen_cfg: dict, bins: list, save_dir: Path = None):
    """
    Create discretize correlation matrix by given conditions, the discretize boundary is customized.
    - data_gen_cfg: dict of data generation config, used to write config on file name
    """
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_data = pd.read_csv(src_dir/f"corr_s{s_l}_w{w_l}.csv", index_col=["item_pair"]).sort_index(axis=1).sort_index(axis=0)
    corr_data.columns = pd.DatetimeIndex(corr_data.columns)
    res_data = corr_data.values.copy()
    num_bins = len(bins)-1
    discretize_idxs = np.digitize(res_data, bins, right=True)
    discretize_idxs[discretize_idxs == 0] = 1
    discretize_idxs[discretize_idxs > num_bins] = num_bins
    discretize_data = discretize_idxs.astype(np.float32)
    all_discretize_values = np.linspace(-1, num_bins-2, num_bins)
    if np.unique(discretize_data).shape[0] != num_bins:
        selected_discretize_values = []
        for i in range(1, num_bins+1):
            if i in np.unique(discretize_data):
                selected_discretize_values.append(all_discretize_values[i-1])
        selected_discretize_values = np.array(selected_discretize_values)
    else:
        selected_discretize_values = all_discretize_values
    assert selected_discretize_values.shape[0] == np.unique(discretize_data).shape[0], f"selected_discretize_values.shape[0]:{selected_discretize_values.shape[0]} != np.unique(discretize_data).shape[0]:{np.unique(discretize_data).shape[0]}"
    for discretize_tag, discretize_value in zip(np.unique(discretize_data), selected_discretize_values):
        discretize_data[discretize_data == discretize_tag] = discretize_value
    discretize_corr_dataset = pd.DataFrame(discretize_data, index=corr_data.index, columns=corr_data.columns)

    LOGGER.info(f"Return discretize_corr_dataset.shape:{discretize_corr_dataset.shape}"
                f"\nThe customized boundary of discretize matrices:\n{bins}"
                f"\nUnique values and correspond counts of discretize_corr_dataset:\n{np.unique(discretize_corr_dataset, return_counts=True)}")
    if save_dir:
        discretize_corr_dataset.to_csv(save_dir/f"corr_s{s_l}_w{w_l}.csv")

    return discretize_corr_dataset


if __name__ == "__main__":
    data_args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    data_args_parser.add_argument("--corr_type", type=str, nargs='?', default="pearson",
                                  choices=["pearson", "cross_corr"],
                                  help="input the type of correlation computing")
    data_args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                                  help="input the number of stride length of correlation computing")
    data_args_parser.add_argument("--corr_window", type=int, nargs='?', default=50,
                                  help="input the number of window length of correlation computing")
    data_args_parser.add_argument("--data_implement", type=str, nargs='?', default="SP500_20082017",  # data implement setting
                                  help="input the name of implemented dataset, watch options by printing /config/data_config.yaml/[\"DATASETS\"].keys()")  # watch options by operate: print(data_cfg["DATASETS"].keys())
    data_args_parser.add_argument("--train_items_setting", type=str, nargs='?', default="train_train",  # train set setting
                                  help="input the setting of training items, options:\n    - 'train_train'\n    - 'train_all'")
    data_args_parser.add_argument("--custom_discrete_bins", type=float, nargs='*', default=[],
                                  help="Decide the custom discrete bins")
    data_args_parser.add_argument("--save_corr_data", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                  help="input --save_corr_data to save correlation data")
    args = data_args_parser.parse_args()
    LOGGER.info(pformat(vars(args), indent=1, width=100, compact=True))

    # generate correlation matrix across time
    DATA_GEN_CFG = {}
    DATA_GEN_CFG['CORR_STRIDE'] = args.corr_stride
    DATA_GEN_CFG['CORR_WINDOW'] = args.corr_window
    DATA_GEN_CFG['CORR_TYPE'] = args.corr_type

    # set directories
    # input & outpu folder settings
    dir_name = DATA_CFG["DATASETS"][args.data_implement]['OUTPUT_FILE_NAME_BASIS'] + "-" + args.train_items_setting
    corr_data_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{dir_name}/{DATA_GEN_CFG['CORR_TYPE']}/corr_data"
    custom_discretize_corr_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{dir_name}"/f"{args.corr_type}"/f"custom_discretize_corr_data/bins_{'_'.join((str(f) for f in args.custom_discrete_bins)).replace('.', '')}"
    corr_data_dir.mkdir(parents=True, exist_ok=True)
    custom_discretize_corr_dir.mkdir(parents=True, exist_ok=True)

    if DATA_CFG["DATASETS"][args.data_implement].get("TRAIN_PAIRS_SET"):
        corr_dataset = set_certain_pairs_corr_data(args.data_implement, DATA_CFG, DATA_GEN_CFG, corr_data_dir, args.save_corr_data)
    elif DATA_CFG["DATASETS"][args.data_implement].get("TRAIN_SET"):
        corr_dataset = set_corr_data(args.data_implement, DATA_CFG, DATA_GEN_CFG, corr_data_dir, args.train_items_setting, args.save_corr_data)
    discretize_corr_dataset = gen_custom_discretize_corr(src_dir=corr_data_dir, data_gen_cfg=DATA_GEN_CFG, bins=args.custom_discrete_bins, save_dir=custom_discretize_corr_dir if args.save_corr_data else None)
