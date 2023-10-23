from pathlib import Path

import pandas as pd
import yfinance as yf

save_dir = Path("../../dataset/raw_data/")
save_file_name = "sp500_hold_19982023_adj_close.csv"

stock_price_20082017_df = pd.read_csv("./stock08_price.csv").set_index("Date")
selected_stock_tickers = list(stock_price_20082017_df.columns)
selected_stock_tickers.remove("SP500")
history = yf.download(tickers=" ".join(selected_stock_tickers), start="1998-01-01", end="2023-10-18", interval="1d")

res_df = history['Adj Close']
res_df.to_csv(save_dir/save_file_name)
