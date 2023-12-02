import argparse
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import rcParams
from pandas._libs.tslibs.timestamps import Timestamp
from ruptures.utils import draw_bkps
from timeseries_generator import (Generator, HolidayFactor, SinusoidalFactor,
                                  WeekdayFactor)
from timeseries_generator.external_factors import (CountryGdpFactor,
                                                   EUIndustryProductFactor)

sys.path.append(str(Path(__file__).parent.parent))
from utils.log_utils import Log

LOGGER = Log().init_logger(logger_name=__name__)
DF_LOGGER = Log().init_logger(logger_name="df_logger")
CURRENT_DIR = Path(__file__).parent


def pw_rand_f1_f2_wavy(n_samples=200, n_bkps=3, seed=120):
    """Return a 1D piecewise wavy signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of changepoints
        seed (int): random seed, the frequence 1 and frequence 2 also based on seed

    Returns:
        tuple: signal of shape (n_samples), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    # we create the signal
    rng = np.random.default_rng(seed=seed)
    f1, f2 = rng.uniform(low=0, high=1, size=(2, 2))
    freqs = np.zeros((n_samples, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n_samples)

    # DeprecationWarning: Calling np.sum(generator) is deprecated
    # Use np.sum(np.from_iter(generator)) or the python sum builtin instead.
    signal = np.sum([np.sin(2 * np.pi * tt * f) for f in freqs.T], axis=0)

    return signal, bkps


def nike_ts(n_samples: int = 200, seed: int = None, start_date: str = "2010/1/1"):
    """Return a time-series with some calender factors(week, holiday...) and external factors.

    Args:
        n_samples (int, optional): signal length
        seed (int, optional): random seed
        start_date (str, optional): string of date

    Returns:
        tuple: signal of shape (n_samples)
    """
    if seed is None:
        country = "United Kingdom"
    else:
        rng = np.random.default_rng(seed=seed)
        country = rng.choice(["United Kingdom", "France", "Germany", "Italy", "Spain"])
    start_date = Timestamp(start_date)
    c_gdp_factor = CountryGdpFactor(country_list=[country])
    eu_industry_product_factor = EUIndustryProductFactor()
    holiday_factor = HolidayFactor(
        holiday_factor=1.5,
        special_holiday_factors={
            "Christmas Day": 3.
        },
        country_list=[country.replace(" ", "_")]
    )
    weekday_factor = WeekdayFactor(
        col_name="weekend_boost_factor",
        factor_values={0: 1.3, 4: 1.15}  # Here we assign a factor of 1.15 to Friday, and 1.3 to Sat/Sun
    )
    product_seasonal_components = SinusoidalFactor(
        feature="basis_trend_seasonality",
        col_name="basis_trend_seasonal_factor",
        feature_values={
            "year": {
                "wavelength": 365.,
                "amplitude": 0.2,
                "phase": 365/4,
                "mean": 1.
            },
        }
    )
    features_dict = {
            "country": [country.lower().replace(" ", "").replace("_", "")],
            "basis_trend_seasonality": ["year"]
        }
    g: Generator = Generator(
        factors={
            c_gdp_factor,
            eu_industry_product_factor,
            holiday_factor,
            weekday_factor,
            product_seasonal_components,
        },
        features=features_dict,
        date_range=pd.date_range(start=start_date, periods=n_samples),
        base_value=10000
    )
    df = g.generate()
    LOGGER.info(f"Generated nike_ts with seed: {seed}, shape: {df.shape}, country: {country}, start_date: {start_date}")

    return df['value'].values


def specific_ts(df_path: Path, ts_name: str):
    df = pd.read_csv(df_path)
    return df[ts_name].values


def gen_leader_signal(args, basis_sig_seed):
    n, n_bkps = args.time_len, args.n_bkps  # time_length(number of samples), number of variables(dimension), number of change points
    rng = np.random.default_rng(seed=0)
    seg_len = int(n/(n_bkps+1))
    n = n-(n % seg_len)  # remove remainder
    basis_m_list = rng.uniform(low=-10, high=10, size=(n_bkps+1, 1))
    basis_b = rng.uniform(low=0, high=10, size=1)
    tt = np.arange(seg_len)
    basis_trend = np.zeros(n)
    bkps = []
    for i, basis_m in enumerate(basis_m_list):
        basis_trend[i*seg_len:(i+1)*seg_len] = basis_m*tt+basis_b
        basis_b = basis_trend[(i+1)*seg_len-1]
        bkps.append((i+1)*seg_len)
    basis_trend_mean = basis_trend.mean()
    if args.basis_type == "pw_rand_wavy":
        wave_signal, _ = pw_rand_f1_f2_wavy(n_samples=n, n_bkps=0, seed=basis_sig_seed)
        leader_signal = (wave_signal*basis_trend_mean)+basis_trend
    elif args.basis_type == "nike_ts":
        nike_signal = nike_ts(n_samples=n, seed=basis_sig_seed)
        leader_signal = nike_signal+basis_trend
    elif args.basis_type == "specific_ts":
        leader_signal = specific_ts(args.specific_ts_path, args.specific_ts_var)

    return leader_signal, bkps


def exec_post_processing(signal: np.array, noise_scale: float):
    rng = np.random.default_rng(seed=0)
    n, dim = signal.shape
    noise_scale = noise_scale/100
    for sub_signal in np.split(signal, dim, axis=1):
        # add noise after create signal
        standard_noise = rng.normal(loc=0, scale=0.5, size=n).reshape(-1, 1)
        scale_noise = noise_scale*sub_signal*standard_noise
        sub_signal += scale_noise
    signal = signal+abs(signal.min())+1
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'var_{i}' for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"

    return signal, df


def gen_power_collection_data(args, collection_idx_seed):
    """
    Generate cluster data whose instances are power ($$X_n = m_n*X_{leader}+b_n$$)(non-linear) correlation to leader_signal
    Specifing `args.n_bkps` to decide the number of change-point of leader_signal.
    """
    dim, noise_scale, n_bkps = args.dim, args.noise_scale, args.n_bkps
    basis_type, power = args.basis_type, args.power
    rng = np.random.default_rng(seed=collection_idx_seed)
    leader_signal, bkps = gen_leader_signal(args, basis_sig_seed=collection_idx_seed)
    no_noise_signal = np.zeros((args.time_len, dim))
    for i, (sub_signal, (reg_coef, reg_bias)) in enumerate(zip(np.split(no_noise_signal, dim, axis=1), rng.uniform(low=-10, high=10, size=(dim, 2)))):
        if i == 0:
            sub_signal += leader_signal.reshape(-1, 1)
        else:
            sub_signal += (reg_coef*(leader_signal**power)+reg_bias).reshape(-1, 1)  # create sub_variable that has linear correlation to power_2 of leader_signal
    signal, df = exec_post_processing(signal=no_noise_signal, noise_scale=noise_scale)
    LOGGER.info(f"Generated power_{power} cluster data with seed: {collection_idx_seed}, shape: {df.shape}, basis_type: {basis_type}, {n_bkps} change points.")
    DF_LOGGER.info(f"Power {power} cluster data[:5, :5]:")
    DF_LOGGER.info(df.iloc[:5, :5])

    return signal, bkps, df


def gen_multi_collections_data(args, gen_data_func):
    """
    Generate multiple cluseter data.
    Construct the clusters by data that produce by `gen_data_func`.
    """
    gen_data_func_name = gen_data_func.__name__
    n, dim = args.time_len, args.dim  # time_length(number of samples), number of variables(dimension)
    n_bkps, sigma = args.n_bkps, args.noise_scale  # number of change points, noise standart deviation
    n_collections = args.n_collections  # number of clusters
    signal = np.zeros((n, n_collections*dim))
    for sub, collection_idx in zip(np.split(signal, n_collections, axis=1), range(n_collections)):
        cluster_signal, _, _ = gen_data_func(args, collection_idx_seed=collection_idx)
        sub += cluster_signal
    dates = pd.to_datetime(range(n), unit='D', origin=pd.Timestamp('now'))  # create a DatetimeIndex with interval of one day
    var_names = [f'cluster_{collection_idx}_var_{i}' for collection_idx in range(n_collections) for i in range(dim)]
    df = pd.DataFrame(signal, index=dates, columns=var_names)
    df.index.name = "Date"
    df_save_file_name = f'bkps{n_bkps}-noise_scale{sigma}.csv'
    LOGGER.info(f"Generated clusters_{n_collections} piecewise {gen_data_func_name} data with shape {df.shape} and {n_bkps} change points.")
    DF_LOGGER.info(f"clusters_{n_collections} piecewise {gen_data_func_name} data[:5, :5]:")
    DF_LOGGER.info(df.iloc[:5, :5])

    return df, df_save_file_name


def set_save_dir(args):
    """
    Set the save directory.
    """
    if args.n_collections:
        base_save_dir = CURRENT_DIR/f'../dataset/is_pre_data/synthetic/dim{args.n_collections*args.dim}/collections_{args.n_collections}/{args.data_type[0]}/basis_{args.basis_type}'
    elif args.specific_ts_path:
        base_save_dir = CURRENT_DIR/f'../dataset/is_pre_data/synthetic/dim{args.dim}/basis_specific_ts/{Path(args.specific_ts_path).stem}'
    else:
        base_save_dir = CURRENT_DIR/f'../dataset/is_pre_data/synthetic/dim{args.dim}/{args.data_type[0]}'
    if args.power:
        save_dir = base_save_dir/f'power_{args.power}'.replace('.', '_')
    else:
        save_dir = base_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate raw data.')
    parser.add_argument('--data_type', type=str, default=['multi_collections'], nargs='+',
                        choices=['power_collection', 'multi_collections'],
                        help='Type of data to generate. (default: pw_constant)')
    parser.add_argument('--time_len', type=int, default=1258, help='Input time length. (default: 1258)')
    parser.add_argument('--dim', type=int, default=5, help='Input dimension(number of variable). (default: 5)')
    parser.add_argument('--noise_scale', type=int, default=5, help='Input noise scale in the form of percentage integer. (default: 5))')
    parser.add_argument('--n_bkps', type=int, default=0, help='Input number of change points. (default: 0)')
    parser.add_argument('--n_collections', type=int, default=0, help='Input number of collections. (default: 0)')
    parser.add_argument('--basis_type', type=str, default='pw_rand_wavy', nargs='?',
                        choices=['nike_ts', 'pw_rand_wavy', 'specific_ts'],
                        help='Type of basis_signal to generate. (default: nike_ts)')
    parser.add_argument('--power', type=float, default=None, help='Input power of gen_power_collection_data().')
    parser.add_argument('--specific_ts_path', type=str, default=None,
                        help='Input path of specific_ts. (default: None)')
    parser.add_argument('--specific_ts_var', type=str, default=None,
                        help='Input variable name of specific_ts. (default: None)')
    parser.add_argument("--save_data", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="input --save_data to save raw data")
    ARGS = parser.parse_args()
    assert bool("power_collection" in ARGS.data_type) == bool(ARGS.power), "`power` should be set when `data_type` contains 'power_collection' and vice versa"
    assert bool("multi_collections" in ARGS.data_type) == bool(ARGS.n_collections >= 2), "`n_collections` should be set when `data_type` is 'multi_collections' and vice versa"
    assert bool("multi_collections" in ARGS.data_type) == bool(len(ARGS.data_type) == 2), "`data_type` should contain another generate data setting when 'multi_collections' is set"
    assert bool("specific_ts" in ARGS.basis_type) == bool(ARGS.specific_ts_path), "`specific_ts_path` should be set when `basis_type` is 'specific_ts'"
    assert bool("specific_ts" in ARGS.basis_type) == bool(ARGS.specific_ts_var), "`specific_ts_var` should be set when `basis_type` is 'specific_ts'"

    if 'multi_collections' in ARGS.data_type:
        ARGS.data_type.remove('multi_collections')
        func = locals()[f'gen_{ARGS.data_type[0]}_data']
        save_df, df_save_file_name = gen_multi_collections_data(args=ARGS, gen_data_func=gen_power_collection_data)
    if ARGS.save_data:
        save_dir = set_save_dir(args=ARGS)
        save_path = save_dir/f'{df_save_file_name}'
        save_df.to_csv(save_path, index=True)
        LOGGER.info(f"Save data to {save_path}")
