import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import json
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FNDNet(nn.Module):
    def __init__(self, embed_size, vocab, vocab_dim, hidden_size=128, p=0.2):
        super().__init__()
        # GloVe Embeddings
        self.glove_emb = nn.Embedding.from_pretrained(vocab.vectors)

        # Parallel Convolution and Maxpool
        self.conv1 = nn.Conv1d(in_channels=vocab_dim, out_channels=hidden_size, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=vocab_dim, out_channels=hidden_size, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=vocab_dim, out_channels=hidden_size, kernel_size=5)
        self.maxpool = nn.MaxPool1d(5)

        # Post-Concatenation Conv-Maxpool
        self.conv4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5)
        self.conv5 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5)
        self.maxpool5 = nn.MaxPool1d(30)

        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(p)
        self.linear1 = nn.Linear(hidden_size*3, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 3)

        self.relu = nn.ReLU()



    def forward(self, article_text):
        embed = self.glove_emb(article_text)
        embed_t = embed.permute(0, 2, 1)

        conv1 = self.conv1(embed_t)
        conv2 = self.conv2(embed_t)
        conv3 = self.conv3(embed_t)

        maxpool1 = self.relu(self.maxpool(conv1))
        maxpool2 = self.relu(self.maxpool(conv2))
        maxpool3 = self.relu(self.maxpool(conv3))

        concat = torch.cat((maxpool1, maxpool2, maxpool3), dim=2)

        conv4 = self.conv4(concat)
        maxpool4 = self.relu(self.maxpool(conv4))
        conv5 = self.conv5(maxpool4)
        maxpool5 = self.relu(self.maxpool5(conv5))

        flat = self.flat(maxpool5)

        linear1 = self.dropout(self.linear1(flat))
        out = self.dropout(self.linear2(linear1))

        print("out size: ", out.size())

        



