import torch
import torch.nn as nn
import sys

sys.path.append("../..")
sys.path.append("..")
import numpy as np
from dataset.convert_data_to_tsv import convert_data_to_tsv
from dataset.rppg_dataset import create_dataloader
from model.rppg_model import RPPGModel
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from config.config import Config

config_ = Config().config

if __name__ == "__main__":
    NUM_EPOCH = int(config_["model"]["num_epochs"])
    LEARNING_RATE = float(config_["model"]["learning_rate"])
    DEVICE = torch.device(config_["device"])
    BATCH_SIZE = int(config_["model"]["batch_size"])
    # Bunu bir kere çalıştırın. rppg_data.tsv varsa elinizde doluysa çalıştırmanıza gerek yok.
    # convert_data_to_tsv(DEVICE)
    train_loader, test_loader = create_dataloader()
    rppg_model = RPPGModel(1, 500, BATCH_SIZE, n_layers=1, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(rppg_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_losses = np.zeros(NUM_EPOCH)
    test_losses = np.zeros(NUM_EPOCH)
    saved_models_dir_ = Path("../inference")
    model_name = "rppg_model"

    if (saved_models_dir_.absolute() / f'{model_name}.pth').exists():
        print(f"Loading Pretrained Model...: {saved_models_dir_.absolute() / f'{model_name}.pth'}")
        saved_model = torch.load(str(saved_models_dir_.absolute() / f'{model_name}.pth'))
        epoch = saved_model["epoch"]
    else:
        saved_model = {}
        epoch = 0
        train_loss = []
        test_loss = []

    if saved_model:
        print("Loading Pretrained Model States...")
        rppg_model.load_state_dict(saved_model['rppg_model'])
        optimizer.load_state_dict(saved_model['optimizer'])
        train_loss = saved_model['train_loss']
        test_loss = saved_model['test_loss']

    for epoch in range(NUM_EPOCH):
        rppg_model.train()
        t0 = datetime.now()

        for inputs, targets in train_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = rppg_model(inputs)
            loss = criterion(outputs, targets.float())

            # Backward and Optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_losses[epoch] = np.mean(train_loss)

        rppg_model.eval()
        for inputs, targets in test_loader:
            outputs = rppg_model(inputs)
            loss = criterion(outputs, targets.float())
            test_loss.append(loss.item())

        test_losses[epoch] = np.mean(test_loss)

        dt = datetime.now() - t0
        print(
            f"Epoch: {epoch + 1} / {NUM_EPOCH}, Train Loss: {train_losses[epoch]:.4f}, Test Loss: {test_losses[epoch]:.4f}, Duration: {dt}")

        if epoch % 100 == 0:
            saved_model['epoch'] = epoch
            saved_model['rppg_model'] = rppg_model.state_dict()
            saved_model['optimizer'] = optimizer.state_dict()
            saved_model['train_loss'] = train_loss
            saved_model['test_loss'] = test_loss
            torch.save(saved_model, str(saved_models_dir_ / f'{model_name}.pth'))
            print(f"Model saved epoch: {epoch}")

        elif epoch % 10000 == 0:
            break

        scheduler.step(test_losses[epoch])

        n_correct = 0.
        n_total = 0.

        for inputs, targets in train_loader:
            # Forward Pass
            outputs = rppg_model(inputs)

            # Get prediction
            predictions = (outputs > 0)

            # update counts
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

        train_acc = n_correct / n_total

        n_correct = 0.
        n_total = 0.

        for inputs, targets in test_loader:
            # Forward Pass
            outputs = rppg_model(inputs)

            # Get prediction
            predictions = (outputs > 0)

            # update counts
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

        test_acc = n_correct / n_total
        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()
