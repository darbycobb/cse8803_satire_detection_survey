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
from torchtext.data.utils import get_tokenizer 
from torchtext import vocab

device = 'cuda'

class ArticleDataset(Dataset):
    def __init__(self):
        self.max_seq_len = 300
        # Load Data
        self.data_path = '../../data/clean_final_data_no_date.csv'

        self.data = pd.read_csv(self.data_path)

        self.labels = self.data['label']
        print(self.labels.unique())
        print(len(self.data.dropna()))
        self.text = self.data['Body']
        counter = Counter()
        for body in self.data.Body:
            counter.update(body.split())
        '''with open('vocab.csv', encoding='utf-8-sig', mode='w') as fp:
            fp.write('KMC,freq\n')  
            for tag, count in counter.items():  
                fp.write('{},{}\n'.format(tag, count))''' 

        glove = vocab.GloVe(name="6B", dim=100, max_vectors=50000)
        self.vocab = vocab.Vocab(counter, min_freq=3, specials=['<unk>', '<pad>'],
                            vectors=glove)

        self.tokenizer = get_tokenizer("basic_english")


    def __getitem__(self, index):
        text = self.data.iloc[index].loc['Body']
        label_map = {'real' : 0, 'satire' : 1, 'fake' : 2}
        label = label_map[self.data.iloc[index].loc['label']]
        # Pad if needed
        tokenized = self.tokenizer(text)
        if len(tokenized) > self.max_seq_len:
            tokenized = tokenized[:self.max_seq_len]
        if len(tokenized) < self.max_seq_len:
            tokenized += ['<pad>'] * (self.max_seq_len - len(tokenized))

        idxs = [self.vocab.stoi[w] for w in tokenized]

        return (torch.tensor(idxs), torch.tensor(label))


    def __len__(self):
        return len(self.data)

    def text_from_idxs(self, text):
        return ' '.join([self.vocab.itos[w] for w in text])