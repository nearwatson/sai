import logging
import torch
import itertools, os, re, sys, time
import torch.nn as nn
import numpy as np
from functools import partial
from collections import namedtuple
from fields import Vocab, Field, Parms, Semantic
from log_conf import logg_process, log_start
from numpy import math


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


    
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
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