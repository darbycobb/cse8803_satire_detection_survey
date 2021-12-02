import os
import torch
import time
import pickle
import argparse
import json
import re
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset import ArticleDataset
from sklearn.model_selection import train_test_split
from cbow import BOW
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import random

def collate_fn_attn(batch):
	return tuple(zip(*batch))


if __name__ == '__main__':	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
    batch_size = 128

    epochs = 5

	# SET SEED
    seed = np.random.seed(100)

    articles = ArticleDataset()

    train_indices, test_indices, _, _ = train_test_split(
    range(len(articles)),
    articles.labels,
    stratify=articles.labels,
    test_size=0.2,
    random_state=seed
    )

    # generate subset based on indices
    train_split = Subset(articles, train_indices)
    test_split = Subset(articles, test_indices)

    # create batches
    train_batches = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    test_batches = DataLoader(test_split, batch_size=batch_size, shuffle=True)

    model = BOW(vocab_size=len(articles.token2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
	# TRAIN
    for epoch in range(epochs):
        progress_bar = tqdm(train_batches, leave=False)
        losses = []
        total = 0
        for inputs, target in progress_bar:
            #print(inputs.shape)
            model.zero_grad()

            #print(target)

            output = model(inputs)
            loss = criterion(output.squeeze(), target)

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {loss:.3f}')

    torch.save(model, 'saved_models/bow_heading_body.pt')
    #model = torch.load('fndnet_1.pt')
    results = []
    # TEST
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(test_batches, leave=False)
        for inputs, target in progress_bar:
            for text, label in zip(inputs, target):
                        outputs = model(text)
                        # Get predicted class
                        prediction = np.argmax(outputs)

                        results.append((label, prediction))

                    
    results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])

    results_df.to_csv('saved_results/test_results_heading_body.csv', index=False)