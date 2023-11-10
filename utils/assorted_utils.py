import logging
from pathlib import Path

import dynamic_yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

from .log_utils import Log

LOGGER = Log().init_logger(logger_name=__name__)

def stl_decompn(corr_series: "pd.Series", overview: bool = False) -> (float, float, float):
    output_resid = 100000
    output_trend = None
    output_period = None
    corr_series.name = corr_series.iloc[0]
    corr_series = corr_series.iloc[1:]
    for p in range(2, 11):
        decompose_result_mult = seasonal_decompose(corr_series, period=p)
        resid_sum = np.abs(decompose_result_mult.resid).sum()
        if output_resid > resid_sum:
            output_resid = resid_sum
            output_trend = decompose_result_mult.trend.dropna()
            output_period = p

    reg = LinearRegression().fit(np.arange(len(output_trend)).reshape(-1, 1), output_trend)

    if overview:
        decompose_result_mult = seasonal_decompose(corr_series, period=output_period)
        trend = decompose_result_mult.trend.dropna().reset_index(drop=True)
        plt.figure(figsize=(7, 1))
        plt.plot(trend)
        plt.plot([0, len(trend)], [reg.intercept_, reg.intercept_+ len(trend)*reg.coef_])
        plt.title("trend & regression line")
        plt.show()
        plt.close()
        decompose_result_mult_fig = decompose_result_mult.plot()
        decompose_result_mult_fig .set_size_inches(10, 12, forward=True)
        for ax in decompose_result_mult_fig.axes:
            ax.tick_params(labelrotation=60) # Rotates X-Axis Ticks by 45-degrees
        plt.show()
        plt.close()

    return output_period, output_resid, output_trend.std(), reg.coef_[0]


def calc_corr_ser_property(corr_dataset: pd.DataFrame, corr_property_df_path: Path):
    """
    Produce property of correlation series in form of dataframe
    """
    corr_property_df_dir = corr_property_df_path.parent
    corr_property_df_dir.mkdir(parents=True, exist_ok=True)
    if corr_property_df_path.exists():
        corr_property_df = pd.read_csv(corr_property_df_path).set_index("items")
        LOGGER.info(f"corr_property_df exists, corr_property_df loaded from {corr_property_df_path}")
    else:
        corr_mean = corr_dataset.mean(axis=1)
        corr_std = corr_dataset.std(axis=1)
        corr_stl_series = corr_dataset.apply(stl_decompn, axis=1)
        corr_stl_array = [[stl_period, stl_resid, stl_trend_std, stl_trend_coef] for stl_period, stl_resid, stl_trend_std, stl_trend_coef in corr_stl_series.values]
        corr_property_df = pd.DataFrame(corr_stl_array, index=corr_dataset.index)
        corr_property_df = pd.concat([corr_property_df, corr_mean, corr_std], axis=1)
        corr_property_df.columns = ["corr_stl_period", "corr_stl_resid", "corr_stl_trend_std", "corr_stl_trend_coef", "corr_ser_mean", "corr_ser_std"]
        corr_property_df.index.name = "items"
        corr_property_df.to_csv(corr_property_df_path)
    return corr_property_df


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
    pipeline_corr_data_dir, corr_dir, target_dir, corr_property_dir, cliques_dir = load_dirs(data_implement=data_implement,
                                                                retrieve_items_setting=retrieve_items_setting,
                                                                corr_type=corr_type, target_df_bins=target_df_bins,
                                                                w_l=w_l, s_l=s_l,
                                                                corr_ser_clac_method=corr_ser_clac_method)
    _, corr_dir, target_dir, corr_property_dir, _ = load_dirs(data_implement=data_implement,
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

    LOGGER.info(f"len(items_implement): {len(items_implement)} and len(all_set): {len(all_set if all_set else [])} and len(train_set): {len(train_set if train_set else [])}")
    LOGGER.info(f"dataset_df.shape:{dataset_df.shape}, corr_df.shape:{corr_df.shape}, target_df.shape:{target_df.shape}")
    LOGGER.info(f"================ In {output_file_name}-corr_s{s_l}_w{w_l} and corr_ser_clac_method:{corr_ser_clac_method} ===============")
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

    return pipeline_corr_data_dir, corr_dir, target_dir, corr_property_dir, cliques_dir
