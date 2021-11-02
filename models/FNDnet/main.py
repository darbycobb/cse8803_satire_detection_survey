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
from fndnet import FNDNet
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

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
    train_batches = DataLoader(train_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_attn)
    test_batches = DataLoader(test_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_attn)
	
    '''model = FNDNet(embed_size=1000, vocab=articles.vocab, vocab_dim=100)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
	# TRAIN
    for epoch in range(epochs):
        epoch_loss_history = []
        with tqdm(train_batches, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")  
                model.train()

                text = torch.stack(batch[0])
                label = torch.stack(batch[1])
                
                # Zero the gradients.
                optimizer.zero_grad()

                # Feed forward
                outputs = model(text)

                # Calculate the batch loss.
                loss = criterion(outputs, label)
                epoch_loss_history.append(loss)

                # Backward pass.
                loss.backward()

                # Update the parameters in the optimizer.
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

        train_loss_history.append(epoch_loss_history)

    torch.save(model, 'fndnet.pt')'''
    model = torch.load('fndnet_1.pt')
    results = []
    # TEST
    model.eval()
    with torch.no_grad():
        with tqdm(test_batches, unit='batch') as tepoch:
            for batch in tepoch:
                    all_text = torch.stack(batch[0])
                    all_labels = torch.stack(batch[1])
                    for text, label in zip(all_text, all_labels):
                        reshaped_text = text[None, :]
                            
                        # Feed forward
                        outputs = model(reshaped_text)
                        # Get predicted class
                        prediction = np.argmax(outputs)

                        text_str = articles.text_from_idxs(text)
                        results.append((text_str, label, prediction))

                    

    results_df = pd.DataFrame(results, columns=['Text', 'Actual', 'Predicted'])

    results_df.to_csv('test_results.csv', index=False)


    '''plt.figure()
    epoch_idxs = range(len(train_loss_history))

    plt.plot(epoch_idxs, train_loss_history, "-b")
    plt.title("Loss")
    #plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.xticks(np.arange(0, max(epoch_idxs)+1, step=1))
    plt.show()'''

    