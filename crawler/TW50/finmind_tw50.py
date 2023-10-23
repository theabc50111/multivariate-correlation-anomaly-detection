import os
import traceback
import sys
import pandas as pd
from tqdm import tqdm
from FinMind.data import DataLoader

finmind_token = os.environ['finmind_token_ywt']
api = DataLoader()
api.login_by_token(api_token=finmind_token)

with open("./tw50_stocks_code_list.txt", "r") as f:
    stocks_list = list(filter(lambda x: x if x else False, f.read().split("\n")))

stock_info_df = api.taiwan_stock_info()
res_df = pd.DataFrame()

for _, stock_code in tqdm(enumerate(stocks_list)):
    try:
        download_df = api.taiwan_stock_daily_adj(
            stock_id=stock_code,
            start_date='2008-01-01',
            end_date='2018-03-13'
        )

        tmp_df = download_df.loc[:, ['date', 'close']]
        stock_name = stock_info_df.loc[stock_info_df["stock_id"] == stock_code, "stock_name"].drop_duplicates().values[0]
        tmp_df = tmp_df.rename(columns={"close": stock_name+"_adj_close"})

        if _ == 0:
            res_df = tmp_df
        else:
            res_df = pd.merge(res_df, tmp_df, on="date", how="left")
    except KeyError:
        with open("finmind_tw50_log.txt", "a") as f:
            f.write(f"{stock_code}/{stock_name} : No adj_close\n")
    except Exception as e:
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        last_call_stack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        file_name = last_call_stack[0]  # 取得發生的檔案名稱
        line_num = last_call_stack[1]  # 取得發生的行號
        func_name = last_call_stack[2]  # 取 得發生的函數名稱
        err_msg = "File \"{}\", line {}, in {}: [{}] {}".format(file_name, line_num, func_name, error_class, detail)
        with open("finmind_tw50_log.txt", "a") as f:
            f.write(f"{stock_code}/{stock_name} : {err_msg} \n")

 
res_df = res_df.rename(columns={"date": "Date"})
res_df.to_csv("./tw50_hold_20082018_adj_close.csv", index=False)
