import jieba, json, os, re, sys, time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn

from nlp_db import nlp_db
from utils import *


class Parms():
    def __init__(self, ):
        self.max_enc_num = 50
        self.max_dec_num = 50
        self.path = "./data/multi30k/"
        self.modes = ['train', 'val', 'test2016']
        self.exts = ['.en.atok', '.de.atok']
        self.ndev = 1
        self.batch_size = 64
        self.batch_size = 5
        self.n_sent = 5
        self.vocab_path = ''


class Semantic():
    """
    special token such as sos, eos and etc
    """
    def __init__(self, ):
        self.PAD_TOKEN = '<pad>'
        self.INIT_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        self.UNK_TOKEN = '<unk>'


args = Parms()
semantic = Semantic()


class Vocab():
    def __init__(self, semantic=None):
        self.semantic = semantic
        self.vocab_list = []
        self.vocab_dict = {}
        if semantic:
            self.vocab_sys = {k: v for k, v in semantic.__dict__.items()}

    def load(self, vocabPath):

        if self.semantic:
            [self.vocab_list.append(v) for k, v in semantic.__dict__.items()]

        with open(vocabPath, 'r', encoding='utf-8') as f:
            for token in f.readlines():
                self.vocab_list.append(token.strip())

        self.vocab_dict = {v: k for k, v in enumerate(self.vocab_list)}
        self.vocab_rdict = {k: v for k, v in enumerate(self.vocab_list)}

    def __len__(self):
        return len(self.vocab_list)

    @property
    def vocab_len(self):
        return len(self.vocab_list)

    @property
    def size(self):
        return len(self.vocab_list)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.vocab_dict:
                return self.vocab_dict[key]
            else:
                return self.vocab_dict['<unk>']
        else:
            return self.vocab_list[key]

    def _make_vocab(self, ):
        pass


vocab = Vocab(semantic)
# vocab.load('./data/multi30k/vocab.txt')

class Field():
    def __init__(self, vocab, preprocess=None, postprocess=None):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.vocab = vocab

    def preprocessing(self, x):
        if self.preprocess:
            #             def head_tail_sent(sent_list):
            #                 return [vocab.vocab_sys['INIT_TOKEN']] + sent_list + [vocab.vocab_sys['EOS_TOKEN']]
            return self.preprocess(x)
        else:
            return x

    def postprocessing(self, x):
        if self.postprocess:
            return self.postprocess(x)
        else:
            return x

    def num_word_id(self, sent_list):
        # suppose string is splited into list then:
        return [vocab[word] for word in sent_list]

    def __call__(self, x):
        return self.postprocessing(self.num_word_id(self.preprocessing(x)))


field_process = Field(vocab, preprocess = lambda sent: sent.strip().split())
# field_process('hello Hebe, where is your husband? \t \n')



def _make_vocab(json_file, vocab_path, thres=2, level='word'):
    word_dict = {}
    with open(json_file, "r", encoding='utf-8') as f:
        for l in f.readlines():
            for token in list(jieba.cut(json.loads(l)['sentence'])):
                if token not in word_dict:
                    word_dict[token] = 0
                else:
                    word_dict[token] += 1

    if not os.path.isfile(vocab_path):
        open(vocab_path,'a').close()

    with open(vocab_path, 'w') as f:
        for k, v in word_dict.items():
            if v > thres:
                print(k, file=f)

# _make_vocab(json_file, vocab_path = args.vocab_path, thres=2)


def _make_chatbot_vocab(file, vocab_path, thres = 2):
    word_dict = {}
    with open(file, "r", encoding='utf-8') as f:
        cnt = 0
        for l in f.readlines():
            for token in list(jieba.cut(l.strip().replace('\t',""))):
                if token not in word_dict:
                    word_dict[token] = 0
                else:
                    word_dict[token] += 1

    if not os.path.isfile(vocab_path):
        open(vocab_path,'a').close()

    with open(vocab_path, 'w') as f:
        for k, v in word_dict.items():
            if v > thres:
                print(k, file=f)
                
                
def get_max_sent_len(file):
    with open(file, "r", encoding='utf-8') as f:
        maxlen, sent_count = 0, 0
        for l in f.readlines():
            maxlen = max([maxlen, max([len(sent) for sent in l.split()])])
            sent_count += 1
    
    return maxlen, sent_count
