import os
import torch
import time
import pickle
import argparse
import json
import re
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset import ArticleDataset
from sklearn.model_selection import train_test_split
from fndnet import FNDNet

def collate_fn_attn(batch):
	return tuple(zip(*batch))


if __name__ == '__main__':	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
    batch_size = 128

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
	
    model = FNDNet(embed_size=1000, vocab=articles.vocab, vocab_dim=100)
	# TEST MODEL
    for batch in train_batches:
            #for batch_idx, (image, captions, caplens) in enumerate(iter(dataloader)): 
                model.train()

                text = torch.stack(batch[0])

                # Feed forward
                outputs = model(text)

	# FOR TRAINING AND VAL DATASETS ONLY
	#run_training_loop(model, val_loader, lr=args.learning_rate, num_epochs=args.num_epochs)

	# EVALUATE
	#evaluate(model, val_loader)

    