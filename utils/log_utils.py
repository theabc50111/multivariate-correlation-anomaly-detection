import logging
import logging.config
import sys
from pathlib import Path

import yaml


class DebugFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.DEBUG:
            return True
        return False

class Log:
    def __init__(self):
        self.this_file_dir = Path(__file__).parent.absolute()

    def init_logger(self: object, logger_name: str = None, config_path: Path = None, edit_config: dict = None):
        path = config_path if config_path else self.this_file_dir/'../config/log_config.yaml'
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            if edit_config is not None:
                for key, value in edit_config.items():
                    config[key] = value
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
