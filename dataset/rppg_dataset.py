import torch
import pandas
from torch.utils.data import Dataset, DataLoader
from config.config import Config

config_ = Config().config

def create_dataloader():
    batch_size = config_["model"]["batch_size"]
    dataset = RPPGDataset("rppg_data.tsv")
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
        self.data = pandas.read_csv(data_path, sep="\t")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx, 0], device=config_["device"], dtype=torch.float), torch.tensor(
            self.data.iloc[idx, 1], device=config_["device"], dtype=torch.float)
