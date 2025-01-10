from architecture import *
import sys
import torch

model = SentimentAnalyzer()

train_loader, val_loader, tokenizer = prepare_data(sys.argv[1])

train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5)
