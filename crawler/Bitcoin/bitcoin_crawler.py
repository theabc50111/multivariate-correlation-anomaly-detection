import json
import re
from pathlib import Path

import pandas as pd
import requests

url = "https://99bitcoins.com/bitcoin/historical-price/"
save_dir = Path("../../dataset/raw_data/")
save_file_name = "bitcoin_20102022.csv"

res = requests.get(url)

# re.search("chartdata\ =\ \{(?P<price>\"price\"\:\[(\[.*\,.*\])*\])\,(?P<event>\"events\"\:\[.*\])\}", res.text).group("price")  # get price string by re
# re.search("chartdata\ =\ \{(?P<price>\"price\"\:\[(\[.*\,.*\])*\])\,(?P<event>\"events\"\:\[.*\])\}", res.text).group("event")  # get event info by re
price_events = re.search("chartdata\ =\ \{.*\}", res.text).group().lstrip("chartdata = ")
price_events = json.loads(price_events)
price_df = pd.DataFrame(price_events['price'], columns=["Date", "Bitcoin"])
price_df["Date"] = pd.to_datetime(price_df["Date"], unit='ms')
price_df = price_df.set_index("Date")
price_df.to_csv(save_dir/save_file_name)
