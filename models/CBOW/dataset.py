import json
import pandas as pd
import numpy as np
import os
import re
import torch
from collections import Counter

from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer 

from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Code modified from https://github.com/scoutbee/pytorch-nlp-notebooks

class ArticleDataset(Dataset):
    def __init__(self):
        self.max_seq_len = 300
        # Load Data
        self.data_path = '../../data/clean_final_data_no_date.csv'

        df = pd.read_csv(self.data_path)
        
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        self.sequences = self.vectorizer.fit_transform(df.Body.tolist())
        
        label_map = {'real' : 0, 'satire' : 1, 'fake' : 2}
        
        self.labels = df.label.apply(lambda x: label_map[x]).tolist()

        self.token2idx = self.vectorizer.vocabulary_
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
    
    def __getitem__(self, i):
        return self.sequences[i, :].toarray(), self.labels[i]
    
    def __len__(self):
        return self.sequences.shape[0]

    def text_from_idxs(self, text):
        return ' '.join([self.idx2token[w] for w in text])
