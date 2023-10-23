import yfinance as yf
import pandas as pd
from pathlib import Path


save_dir = Path("../../ARIMA-LSTM-hybrid-corrcoef-predict_YWT/extended_reasearch_YWT/dataset/raw_data/")
file_name = "nvda_20102022.csv"

selected_stock_tickers = ['NVDA']
history = yf.download(tickers=" ".join(selected_stock_tickers), start="2010-07-17", end="2022-09-24")
res_df = history['Adj Close'].rename("NVDA")

res_df.to_csv(save_dir/file_name)
