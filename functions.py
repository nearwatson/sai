import jieba, json, os, re, sys, time
from datetime import datetime
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import shutil

from fields import Field, Parms, Semantic, Vocab, _make_vocab
from utils import *

from nlp_db import nlp_db

from model_class import NLU_Classify


def pathFiles(rel_path):
    return [
    os.path.join(os.path.abspath(rel_path),
                 os.listdir(rel_path)[i])
    for i in range(len(os.listdir(rel_path)))
]

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
                
def func_pad(sent, max_sent_len):
    return [vocab.__getitem__(token) for token in jieba.cut(sent)
            ] + [0] * (max_sent_len - len(list(jieba.cut(sent)))) , len(list(jieba.cut(sent)))


def acc(y_hat, y_label):
    correct = (torch.argmax(y_hat, dim = 1) == y_label).float()
    acc_rate = correct.sum() / len(correct)
    
    return acc_rate

def dump_log(manual_log):
    with open(manual_log, 'a') as fp:
        json.dump(
            {
                "epoch": last_epoch,
                "loss": last_loss.data.item(),
                "train_avg_acc": last_avgac.data.item(),
                "dev_avg_acc": dev_acc.data.item()
            }, fp)
        fp.write('\n')
    
    with open(manual_log, 'r') as f:
        last_ten_line = f.readlines()[-10:]
    
    with open(manual_log, 'w') as f:
        for line in last_ten_line:
            f.write(line)
            
def get_last_epoch():
    with open(args.manual_log, 'r') as f:
        l = f.readlines()[-1]
        last_epoch = json.loads(l.strip())['epoch']
        
    return last_epoch