'''
    Preprocess Data
'''
import json
import pandas as pd
import numpy as np
import os
import re
import torch

from torch.utils.data import DataLoader, Dataset
from torchtext import vocab, data

device = 'cuda'

class ArticleDataset(Dataset):
    def __init__(self):
        # Load Data
        data = pd.read_csv('../../../data/final_data.csv', index=False)
        
        # Create Vocabulary
        vec = vocab.Vectors('glove.840B.300d.txt')
        vocabulary = vocab.Vocab(counter, max_size=500000, vectors=vec, specials=['<pad>', '<unk>'])

    def get_body_for_article(self):
        pass

    def get_article(self, article_path):
        pass

    def load_article_from_path(self, path):
        pass

    def __getitem__(self, index):

        caption = self.caps[index]

        words = re.compile(r'\w+')
        tokens = words.findall(caption.lower())
        caption_len = len(tokens) + 2

        if caption_len > self.max_seq_len:
            caption_len = self.max_seq_len
        # Replace words not in vocab with <UNK>
        for i in range(len(tokens)):
            if tokens[i] not in self.vocab:
                tokens[i] = '<UNK>'
        # Add <SOS> at beginning and <EOS> tokens at end
        tokens.insert(0,'<SOS>')
        tokens.insert(len(tokens), '<EOS>')
        # Pad if necessary, shorten if necessary (should only be one caption)
        if len(tokens) < self.max_seq_len:
            for i in range(self.max_seq_len-len(tokens)):
                tokens.append('<PAD>')
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:(self.max_seq_len-1)]
            tokens.append('<EOS>')

        # Map words to indices
        int_tokens = torch.LongTensor([self.vocab_map[i] for i in tokens])

        return (int_tokens, caption_len)


    def __len__(self):
        return len(self.im_names)