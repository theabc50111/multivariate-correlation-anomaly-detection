import json
import re
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

from .log_utils import Log
from .plot_utils import plot_corr_distribution, plot_tr_val_labels_pie

LOGGER = Log().init_logger(logger_name=__name__)
DF_LOGGER = Log().init_logger(logger_name="df_logger")


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
        plt.plot([0, len(trend)], [reg.intercept_, reg.intercept_+len(trend)*reg.coef_])
        plt.title("trend & regression line")
        plt.show()
        plt.close()
        decompose_result_mult_fig = decompose_result_mult.plot()
        decompose_result_mult_fig .set_size_inches(10, 12, forward=True)
        for ax in decompose_result_mult_fig.axes:
            ax.tick_params(labelrotation=60)  # Rotates X-Axis Ticks by 45-degrees
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
        LOGGER.info(f" :  \ncorr_property_df exists, corr_property_df loaded from {corr_property_df_path}")
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


def calc_tr_val_corr_and_labels_distribution(tr_data: dict, val_data: dict, corr_df: pd.DataFrame, data_implement: str, custom_idxs: list = None, plot_distribution: bool = False) -> pd.DataFrame:
    """
    Produce distribution of correlation series and labels in form of dataframe
    """
    if custom_idxs:
        selected_tr_data = {"model_input": tr_data["model_input"][custom_idxs, ::], "target": tr_data["target"][custom_idxs, ::]}
        selected_val_data = {"model_input": val_data["model_input"][custom_idxs, ::], "target": val_data["target"][custom_idxs, ::]}
    else:
        selected_tr_data = tr_data
        selected_val_data = val_data
    assert (np.unique(selected_tr_data['target']).shape[0] < 10) and (np.unique(selected_val_data['target']).shape[0] < 10), "number of classes should lower than 10"
    tr_labels, tr_labels_freq_counts = np.unique(selected_tr_data['target'], return_counts=True)
    val_labels, val_labels_freq_counts = np.unique(selected_val_data['target'], return_counts=True)
    tr_labels_pct = tr_labels_freq_counts/tr_labels_freq_counts.sum()
    val_labels_pct = val_labels_freq_counts/val_labels_freq_counts.sum()
    labels_pct_each_class = {f"tr_class_{label}": pct for label, pct in zip(tr_labels, tr_labels_pct)} | {f"val_class_{label}": pct for label, pct in zip(val_labels, val_labels_pct)}
    data_implement_dict = {"data_implement": [data_implement]}
    corr_df_std = {"std of all corr": [corr_df.values.std()]}
    distribution_df = pd.DataFrame(data_implement_dict | labels_pct_each_class | corr_df_std)
    if plot_distribution:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 18))
        plot_tr_val_labels_pie(tr_labels=tr_labels, val_labels=val_labels, tr_labels_freq_counts=tr_labels_freq_counts, val_labels_freq_counts=val_labels_freq_counts, axes=axes)
        plot_corr_distribution(corr_df=corr_df, axes=axes)
        fig_title = re.sub(r"SP500_\d*_", "", data_implement)
        fig_title = f'{fig_title}_cross_term' if custom_idxs else fig_title
        fig.suptitle(f'{fig_title}', fontsize=40)
        plt.show()
        plt.close()

    return distribution_df


def calc_mix_model_loss_history(log_path_list: list, samples_weights: tuple, loss_history_len: int) -> (np.ndarray, np.ndarray):
    """
    Calculate mix loss history from multiple log files
    """
    assert len(log_path_list) == len(samples_weights), f"len(log_path_list) should be same as len(samples_weights), but len(log_path_list):{len(log_path_list)}, len(samples_weights):{len(samples_weights)}"
    mix_tr_loss_history = np.zeros((loss_history_len,))
    mix_val_loss_history = np.zeros((loss_history_len,))
    for log_path, weight in zip(log_path_list, samples_weights):
        with open(log_path, "r") as source:
            log_dict = json.load(source)
            tr_loss_each_log = np.array(log_dict["tr_loss_history"])*weight
            val_loss_each_log = np.array(log_dict["val_loss_history"])*weight
            mix_tr_loss_history += tr_loss_each_log
            mix_val_loss_history += val_loss_each_log
    mix_tr_loss_history /= sum(samples_weights)
    mix_val_loss_history /= sum(samples_weights)

    return mix_tr_loss_history, mix_val_loss_history


def report_preds_err_degree(model_input_df: pd.DataFrame, tr_val_tt_len_list: list, inference_data: dict, preds: np.ndarray, labels: np.ndarray, data_sp_mode: str, report_save_path: Path) -> None:
    """
    This function is used to report the false predictions of the model.
    """
    num_pairs, tol_time_steps = model_input_df.shape
    assert num_pairs == inference_data["model_input"].shape[0] == inference_data["target"].shape[0], "The number of pairs in the model input dataframe must be equal to the number of pairs in the inference data."
    assert len(tr_val_tt_len_list) == 3, "The length of the `tr_val_tt_list` list must be 3."
    assert sum(tr_val_tt_len_list) == tol_time_steps, "The sum of the `tr_val_tt_list` list must be equal to the total number of time steps in the model input dataframe."
    assert inference_data["target"].shape[1] in tr_val_tt_len_list, "The number of time steps in the inference data target must be in the `tr_val_tt_list` list."
    model_input_dates = model_input_df.columns
    tr_dates_range = (model_input_dates[0:int(tr_val_tt_len_list[0])])
    val_dates_range = (model_input_dates[int(tr_val_tt_len_list[0]):int(tr_val_tt_len_list[0])+int(tr_val_tt_len_list[1])])
    test_dates_range = (model_input_dates[int(tr_val_tt_len_list[0])+int(tr_val_tt_len_list[1]):])
    assert len(tr_dates_range) == tr_val_tt_len_list[0] and len(val_dates_range) == tr_val_tt_len_list[1] and len(test_dates_range) == tr_val_tt_len_list[2], "The length of the dates range must be equal to the length of the `tr_val_tt_len` list."
    data_sp_dates = {"tr": tr_dates_range, "val": val_dates_range, "test": test_dates_range}[data_sp_mode]
    report_df = pd.DataFrame()
    for i, pair_name in enumerate(model_input_df.index):
        labels_each_pair = labels[:, i].astype("int64")
        preds_each_pair = preds[:, i].astype("int64")
        selected_dates = data_sp_dates[-1*len(preds_each_pair):]
        corr_coef_data = model_input_df.loc[pair_name, selected_dates].to_numpy()
        ori_labels = inference_data["target"][i, -1*len(preds_each_pair):].astype("int64")
        corr_coef_t_shift_diff = np.concatenate([[0], np.diff(corr_coef_data)])
        preds_t_shift_diff_each_pair = np.concatenate([[0], np.diff(preds_each_pair)])
        preds_err_degree_each_pair = abs(labels_each_pair-preds_each_pair)
        df_each_pair_idx = pd.MultiIndex.from_product([[pair_name], ["corr_coef", "corr_coef_t_shift_diff", "ori_labels", "new_labels", "preds", "preds_t_shift_diff", "preds_err_degree"]], names=["pair_name", "data_category"])
        df_each_pair = pd.DataFrame([corr_coef_data, corr_coef_t_shift_diff, ori_labels, labels_each_pair, preds_each_pair, preds_t_shift_diff_each_pair, preds_err_degree_each_pair], columns=selected_dates, index=df_each_pair_idx)
        report_df = pd.concat([report_df, df_each_pair], axis=0)
    if report_save_path is not None:
        report_df.to_csv(report_save_path)

    LOGGER.info(f"num_pairs: {num_pairs}, data_sp_mode: {data_sp_mode}, data_sp_dates_range: `{data_sp_dates[0]} ~ {data_sp_dates[-1]}`, selected_dates_range: `{selected_dates[0]} - {selected_dates[-1]}`")
    DF_LOGGER.info(f"report_df:\n{report_df}")
    LOGGER.info("-"*50)


def find_anomalies(model_name: str, model_weights_name: str, data_sp_mode: str, save_report: bool = False) -> None:
    """
    This function is used to find the anomalies of the model.
    """
    THIS_FILE_DIR = Path(__file__).resolve().parent
    report_df_dir = THIS_FILE_DIR/f"../models/exploration_model_result/model_result_csvs/{model_name}/{model_weights_name}/"
    report_df_path = report_df_dir/f"report_preds_err_degree-{data_sp_mode}.csv"
    report_df = pd.read_csv(report_df_path, index_col=['pair_name', 'data_category'])
    pairs = report_df.index.get_level_values('pair_name').unique()
    all_anomalies_info_df = pd.DataFrame()
    for pair in pairs:
        pair_df = report_df.loc[pair, ::]
        idx = pd.MultiIndex.from_tuples(tuple(product([pair], pair_df.index, repeat=1)), names=['pair_name', 'data_category'])
        pair_df.index = idx
        preds_err_mask = pair_df.loc[(pair, "preds_err_degree"), ::] > 0
        preds_change_mask = pair_df.loc[(pair, "preds_t_shift_diff"), ::] > 0
        dates_keep_mask = (preds_err_mask | preds_change_mask).T.tolist()
        selected_dates_df = pair_df.iloc[::, dates_keep_mask]
        if not selected_dates_df.empty:
            to_be_null_values = (selected_dates_df.loc[(pair, ["preds_t_shift_diff", "preds_err_degree"]), :]).replace(0, np.nan, inplace=False)
            selected_dates_df.loc[(pair, ["preds_t_shift_diff", "preds_err_degree"]), :] = to_be_null_values
            all_anomalies_info_df = pd.concat([all_anomalies_info_df, selected_dates_df], axis=0)
            all_anomalies_info_df = all_anomalies_info_df.sort_index(axis=1)
    all_anomalies_info_df_copy = all_anomalies_info_df.copy()
    all_anomalies_preds_change_info_df = all_anomalies_info_df_copy.loc[(slice(None), ["preds_t_shift_diff"]), :]
    all_anomalies_preds_change_info_df = all_anomalies_preds_change_info_df.dropna(thresh=1, axis=1)
    all_anomalies_preds_change_info_df = all_anomalies_preds_change_info_df.dropna(thresh=1, axis=0)
    all_anomalies_preds_err_info_df = all_anomalies_info_df_copy.loc[(slice(None), ["preds_err_degree"]), :]
    all_anomalies_preds_err_info_df = all_anomalies_preds_err_info_df.dropna(thresh=1, axis=1)
    all_anomalies_preds_err_info_df = all_anomalies_preds_err_info_df.dropna(thresh=1, axis=0)
    if save_report:
        all_anomalies_preds_change_info_df.to_csv(report_df_dir/f"all_anomalies_preds_change_info-{data_sp_mode}.csv")
        all_anomalies_preds_err_info_df.to_csv(report_df_dir/f"all_anomalies_preds_err_info-{data_sp_mode}.csv")
        all_anomalies_info_df.to_csv(report_df_dir/f"all_anomalies_info-{data_sp_mode}.csv")
    DF_LOGGER.info(all_anomalies_preds_change_info_df)
    DF_LOGGER.info(all_anomalies_preds_err_info_df)
    DF_LOGGER.info(all_anomalies_info_df)

