import os
import pickle

import torch
from torch.utils.data import Dataset

root_dir = os.path.abspath("..")


# класс датасетов для модели, в котором данные токенизируются
class DescDataset(Dataset):

    def __init__(self, X, y,tokenizer, max_length=1024):
        super().__init__()

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        self.tokenizer = tokenizer
        self.end_token = "<|endoftext|>"
        self.start_token = "<|startoftext|>"
        self.descriptions = []

        for i in range(y.shape[0]):
            self.descriptions.append(
                self.tokenizer(
                    f"{self.start_token}{y[i]}. {X[i][:max_length]}{self.end_token}"))

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        return self.descriptions[item]


def default_config(root_dir=root_dir):
    config = {
        'batch_size': 16,
        'epochs': 8,
        'learning_rate': 3e-5,
        'max_seq_len': 400}

    with open(root_dir + "/model/config", "wb") as fp:
        pickle.dump(config, fp)

    return config
