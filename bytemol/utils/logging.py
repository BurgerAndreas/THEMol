# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Union


def _timestamp_formatter(with_lineno: bool = False) -> logging.Formatter:
    return logging.Formatter(
        fmt='[%(asctime)s PID %(process)d] %(levelname)s %(message)s'
        if not with_lineno else '[%(asctime)s PID %(process)d] %(levelname)s %(message)s \t(%(pathname)s:%(lineno)d)',
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_timestamp_logging(file_path=None, mode='a', level=logging.INFO):
    """Set up timestamp based logging which outputs in the style of
    ``YEAR-MONTH-DAY HOUR:MINUTE:SECOND.MILLISECOND LEVEL MESSAGE``.

    Parameters
    ----------
    file_path: str, optional
        The file to write the log to. If none, the logger will
        print to the terminal.
    """
    formatter = _timestamp_formatter()

    if file_path is None:
        logger_handler = logging.StreamHandler(stream=sys.stdout)
    else:
        logger_handler = logging.FileHandler(file_path, mode=mode)

    logger_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(logger_handler)
    return


def setup_default_logging(stdout: bool = True,
                          *,
                          file_path=None,
                          file_mode='a',
                          level=logging.INFO,
                          formatter: Union[str, logging.Formatter] = 'time'):
    if formatter == 'time':
        formatter = _timestamp_formatter()
    elif formatter == 'lineno':
        formatter = _timestamp_formatter(with_lineno=True)
    elif not isinstance(formatter, logging.Formatter):
        raise ValueError(f'invalid formatter {formatter}')

    logger = logging.getLogger()
    logger.setLevel(level)

    # clear old handlers
    logger.handlers = []

    if stdout:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file_path != None:
        file_handler = logging.FileHandler(file_path, mode=file_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
