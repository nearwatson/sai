import logging
import torch
import itertools, os, re, sys, time
import torch.nn as nn
import numpy as np
from functools import partial
from collections import namedtuple
from fields import Vocab, Field, Parms, Semantic

def logg_process(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Logs
    # logger.debug('A debug message')
    # logger.info('An info message')
    # logger.warning('Something is not right.')
    # logger.error('A Major error has happened.')
    # logger.critical('Fatal error. Cannot continue')

    return logger


def log_start(path, restart=True):
    """
    if restart = Ture then remove the existed log.file and creates a new one
    """
    if restart:
        if os.path.isfile(path):
            os.remove(path)
        logger = logg_process(path)
    else:
        logger = logg_process(path)

    return logger