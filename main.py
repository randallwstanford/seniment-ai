from architecture import *
import torch
import sys

param1 = sys.argv[1]
model = SentimentAnalyzer()
train_loader, val_loader, tokenizer = prepare_data('data/Valid.csv')

model.load_state_dict(torch.load('llm.pt', weights_only=True))
model.eval()

encoding = tokenizer(
    param1,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

with torch.no_grad():
    prediction = model(encoding['input_ids'], encoding['attention_mask'])
    print(f"Predicted sentiment score: {prediction.item()}")
