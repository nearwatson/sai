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

semantic = Semantic()
# vocab = Vocab(semantic)
# path = "./data/multi30k/vocab.txt"

# vocab.load(path)

# logger = log_start('./logfile.log')
# logger.info("vocab.size:", vocab.size)


class NLU(nn.Module):
    def __init__(self, ):
        super(NLU, self).__init__()
        self.type = 'projector'


class NLU_Classify(nn.Module):
    def __init__(self, class_num, vocab):
        super(NLU_Classify, self).__init__()
        self.type = 'classifier'
        self.batch_size = 101
        self.serial_len = 2
        self.emb = nn.Embedding(vocab.size, embedding_dim=128)
        self.lstm = nn.LSTM(128,
                            args.lstm_hid,
                            args.lstm_step_num,
                            batch_first=True)
        self.fc = nn.Linear(64, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # ? not sure serial_len , batch_size is 100% right

        x = self.emb(x)
        h0 = torch.randn(self.serial_len, self.batch_size, args.lstm_hid)
        c0 = torch.randn(self.serial_len, self.batch_size, args.lstm_hid)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc(x)
        x = x[:, -1, :]
        x = self.softmax(x)
        result = x

        return result


# fw = NLU_Classify(class_num=args.class_num, vocab=vocab)

# result = fw(batch)

# result.shape

# shape seems to be right , yet number should be



# ut = NLU_Classify(10)


# sent_inds = [np.random.randint(vocab.size, size=20) for i in range(args.batch_size)]
# ## sent must of the same size for embed

# inputs = torch.tensor(sent_inds)
# output = ut(inputs)

# # %cat ./logfile.log

# for sent in output:
#     a = [vocab.vocab_list[ind] for ind in sent]
#     print(a)