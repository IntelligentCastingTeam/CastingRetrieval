import logging
from functools import wraps
from logging import handlers

from vine import wrap


def logger(func):
    """为python内置的logging 添加日期滚动存储"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
            handlers=[
                handlers.TimedRotatingFileHandler(
                    filename="./logs/experiments.log", when="D", encoding="utf-8"
                ),
                # logging.StreamHandler(),
            ],
        )
        return func(*args, **kwargs)

    return wrapper
