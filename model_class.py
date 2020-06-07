import logging
import torch
import itertools, os, re, sys, time
import torch.nn as nn
import numpy as np
from functools import partial
from collections import namedtuple
from fields import Vocab, Field, Parms, Semantic
from log_conf import logg_process, log_start


args = Parms()
args.lstm_step_num = 2
args.lstm_hid = 64
semantic = Semantic()
vocab = Vocab(semantic)
path = "./data/multi30k/vocab.txt"

vocab.load(path)

logger = log_start('./logfile.log')
logger.info("vocab.size:", vocab.size)


class NLU(nn.Module):
    def __init__(self, ):
        super(NLU, self).__init__()
        self.type = 'projector'


class NLU_Classify(nn.Module):
    def __init__(self):
        super(NLU_Classify, self).__init__()
        self.type = 'classifier'
        self.emb = nn.Embedding(vocab.size, embedding_dim=128)
        self.lstm = nn.LSTM(128, args.lstm_hid, args.lstm_step_num)
        #         , batch_first = True)
        self.fc = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.emb(x)
        h0 = torch.randn(2, 20, args.lstm_hid)
        c0 = torch.randn(2, 20, args.lstm_hid)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc(x)
        x = self.softmax(x)
        result = torch.argmax(x, dim=1)
        # logger.info('inputs shape{}'.format(x.shape))
        # logger.info('emb shape {}'.format(x.shape))
        # logger.info('lstm shape {}'.format(x.shape))
        # logger.info('fc shape {}'.format(x.shape))
        # logger.info('softmax shape{}'.format(x.shape))
        # logger.info('result:'.format(result))

        return result



ut = NLU_Classify()


sent_inds = [np.random.randint(vocab.size, size=20) for i in range(args.batch_size)]
## sent must of the same size for embed

inputs = torch.tensor(sent_inds)
output = ut(inputs)

# %cat ./logfile.log

for sent in output:
    a = [vocab.vocab_list[ind] for ind in sent]
    print(a)