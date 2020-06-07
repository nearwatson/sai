import os, re, sys, time
from datetime import datetime

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


class Semantic():
    def __init__(self, ):
        self.INIT_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        self.PAD_TOKEN = '<pad>'
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
            return self.vocab_lst[key]

    def _make_vocab(self, ):
        pass


vocab = Vocab(semantic)
vocab.load('./data/multi30k/vocab.txt')



# vocab.vocab_sys, vocab.vocab_dict
# vocab['self']    2731


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


field_process = Field(vocab, preprocess=lambda sent: sent.strip().split())

field_process('hello Hebe, where is your husband? \t \n')
