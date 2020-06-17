from torchtext.data.utils import get_tokenizer
import jieba
import json
import os
import re
import sys
import shutil
import time
from datetime import datetime
import numpy as np
import tensorboardX

import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

from fields import Field, Parms, Semantic, Vocab, _make_vocab
from utils import *

from nlp_db import nlp_db

from model_class import NLU_Classify, TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

semantic = Semantic()
args = Parms()
vocab = Vocab(semantic)

args.manual_log = './manualog_transfromer1.log'
args.model_path = './model_stores/transformer_wiki1.pth'

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)



ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value

predict_model = TransformerModel(
    ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
predict_model.load_state_dict(torch.load(args.model_path))


def data2sent(data, func=None):
    if func:
        return [[TEXT.vocab.itos[func(ind).data.item()] for ind in data[:, i]]
                for i in range(data.shape[1])]

    else:
        return [[TEXT.vocab.itos[ind.data.item()] for ind in data[:, i]]
                for i in range(data.shape[1])]


# data2sent(data)


# data2sent(predict_model(data[1:10, :]), func=lambda word_tensor: torch.argmax(
#     word_tensor, dim=-1))


########################################################################################################
bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
    
def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# Get Results
bptt = 35
i = bptt * 2
# data, targets = get_batch(test_data, i)
data, targets = get_batch(train_data, i)
# best_model.eval()
output = predict_model(data)
########
#############################################################################################################
