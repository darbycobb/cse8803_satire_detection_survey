import torch
from torch import nn

import torchtext

from collections import defaultdict
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

# Code from https://nbviewer.org/github/scoutbee/pytorch-nlp-notebooks/blob/develop/1_BoW_text_classification.ipynb

class BOW(nn.Module):
    def __init__(self, vocab_size, hidden1=128, hidden2=64):
        super(BOW, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 3)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self.relu(self.fc1(inputs.squeeze(1).float()))
        x = self.relu(self.fc2(x))
        return self.fc3(x)