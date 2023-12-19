import logging
import logging.config
import sys
from collections import deque
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import yaml


class DebugFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.DEBUG:
            return True
        return False

class Log:
    def __init__(self, df_max_rows: int = 10, df_max_columns: int = 10, expand_frame_repr: bool = False):
        self.this_file_dir = Path(__file__).parent.absolute()
        self.max_rows = df_max_rows
        self.max_columns = df_max_columns
        self.expand_frame_repr = expand_frame_repr
        self.set_pd_display()

    def init_logger(self: object, logger_name: str = None, config_path: Path = None, update_config: List[Tuple[deque, Any]] = None):
        path = config_path if config_path else self.this_file_dir/'../config/log_config.yaml'
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            if update_config is not None:
                for keys, value in update_config:
                    extracted_config = Log.extract_dict_from_nested_dict(config, keys)
                    extracted_config[keys[-1]] = value
            logging.config.dictConfig(config)
            logger = logging.getLogger(logger_name)
            return logger

    @staticmethod
    def get_ancestors(logger: logging.Logger):
        ancestors = []
        while logger.parent:
            ancestors.append(logger.parent.name)
            logger = logger.parent
        return ancestors

    @staticmethod
    def extract_dict_from_nested_dict(nested_dict: dict, keys: deque):
        if len(keys) == 1:
            return nested_dict
        else:
            key = keys.popleft()
            return Log.extract_dict_from_nested_dict(nested_dict[key], keys)

    def set_pd_display(self):
        pd.set_option('display.max_rows', self.max_rows)
        pd.set_option('display.max_columns', self.max_columns)
        pd.set_option('display.expand_frame_repr', self.expand_frame_repr)
