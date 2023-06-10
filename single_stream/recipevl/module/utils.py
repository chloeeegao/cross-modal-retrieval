import logging
from logging import FileHandler
import os
import sys
import nltk

def setup_logger(name, save_dir, filename):
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    filename = os.path.join(save_dir, filename)
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
        
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_token_ids(sentence, vocab):
    tok_ids = []
    tokens = nltk.tokenize.word_tokenize(sentence.lower())
    tok_ids.append(vocab['<start>'])
    for token in tokens:
        if token in vocab:
            tok_ids.append(vocab[token])
        else:
            tok_ids.append(vocab['<unk>'])
    tok_ids.append(vocab['<end>'])
    return tok_ids
