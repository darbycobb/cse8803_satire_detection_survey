'''
    Preprocess Data
'''
import json
import pandas as pd
import numpy as np
import os
import re
import torch
from collections import Counter

from torch.utils.data import DataLoader, Dataset
from torchtext import vocab, data

device = 'cuda'

class ArticleDataset(Dataset):
    def __init__(self):
        self.max_seq_len = 200
        # Load Data
        self.data_path = '../../../data/final_data_combined.csv'
        self.data = pd.read_csv(data_path, index=False)
        self.labels = data.loc['label']
        self.text = data.loc['Heading_Body']

        # Create Vocabulary
        counter = Counter()
        for comment in data.Heading_Body:
            counter.update(comment.split())
        
        vec = vocab.Vectors('glove.840B.300d.txt')
        vocabulary = vocab.Vocab(counter, max_size=500000, vectors=vec, specials=['<pad>', '<unk>'])
        torch.zero_(vocabulary.vectors[1]); # fill <unk> token as 0

    def get_body_for_article(self):
        pass

    def get_article(self, article_path):
        pass

    def load_article_from_path(self, path):
        pass

    def __getitem__(self, index):

        text = self.data.iloc[index].loc['Heading_Body']

        words = re.compile(r'\w+')
        tokens = words.findall(text.lower())
        text_len = len(tokens) + 2

        if text_len > self.max_seq_len:
            text_len = self.max_seq_len
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
        return len(self.data)