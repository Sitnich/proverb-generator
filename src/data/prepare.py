import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

import src.models.model_classes as cl

root_dir = os.path.abspath("..")


def prepare_data(root_dir=root_dir):
    df = pd.read_csv(root_dir + '/data/proverbs.csv')
    df = df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df['Poverb'], df['Tag'], test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test


def prepare_dataloader(X, y):
    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    dataset = cl.DescDataset(X, y, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return data_loader


def download_model(root_dir=root_dir):
    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    model = AutoModelForCausalLM.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    torch.save(model.state_dict(), root_dir + '/model/rugpt2_proverb_raw_dot.pt')
    torch.save(tokenizer, root_dir + '/model/rugpt2_proverb_token.pt')
