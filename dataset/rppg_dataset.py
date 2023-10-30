import os.path

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config.config import Config

config_ = Config().config


def create_dataloader():
    batch_size = int(config_["model"]["batch_size"])
    dataset = RPPGDataset("rppg_data.pkl")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


class RPPGDataset(Dataset):
    def __init__(self, data_path):
        assert os.path.exists(data_path)
        self.data = pd.read_pickle(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(np.array(self.data.iloc[idx, 0])[..., np.newaxis], device=config_["device"],
                            dtype=torch.float), torch.tensor(
            self.data.iloc[idx, 1].astype(np.int32)[..., np.newaxis], device=config_["device"], dtype=torch.int32)
