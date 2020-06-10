import jieba, json, os, re, sys, time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn

def read_json(file, thresh=20, k=None, func=None):
    with open(file, "r", encoding='utf-8') as f:
        rzlt = []
        cnt = 0
        for l in f.readlines():

            if k != None and func != None:
                rzlt.append(func(json.loads(l)[k]))

            elif k != None:
                rzlt.append(json.loads(l)[k])

            else:
                rzlt.append(json.loads(l))

            if cnt > thresh:
                break

    return rzlt



def json_iter(file, batch_size=1000, k=None, func=None):
    with open(file, "r", encoding='utf-8') as f:
        rzlt = []
        for l in f.readlines():
            if k != None and func != None:
                rzlt.append(func(json.loads(l)[k]))

            elif k != None:
                rzlt.append(json.loads(l)[k])

            else:
                rzlt.append(json.loads(l))

            if len(rzlt) == batch_size:

                yield rzlt
                rzlt = []



def restart_iter(batch_size):
    x_iter = json_iter(file=cfiles[2],
                       batch_size=batch_size,
                       k='sentence',
                       func=lambda sent:
                       [vocab.__getitem__(token)
                        for token in jieba.cut(sent)] + [0] *
                       (max_sent_len - len(list(jieba.cut(sent)))))

    y_iter = json_iter(file=cfiles[2],
                       batch_size = batch_size,
                       k='label',
                       func=lambda x: label_rdict[x])

    return x_iter, y_iter



def data_iter(data, batch_size):
    i = int(batch_size)

    while i < len(data):
        yield data[i - batch_size:i]
        i += batch_size