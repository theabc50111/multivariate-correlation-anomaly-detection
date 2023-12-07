import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

from .log_utils import Log
from .plot_utils import plot_corr_distribution, plot_tr_val_labels_pie

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
