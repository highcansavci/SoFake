import torch
import torch.nn as nn
import numpy as np
from dataset.convert_data_to_tsv import convert_data_to_tsv
from dataset.rppg_dataset import create_dataloader
from model.rppg_model import RPPGModel
from datetime import datetime
import matplotlib.pyplot as plt

NUM_EPOCH = 1000
LEARNING_RATE = 1e-4
DEVICE = torch.device("cpu")
convert_data_to_tsv(DEVICE)
train_loader, test_loader = create_dataloader()
rppg_model = RPPGModel(300, 100, 10, DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(rppg_model.parameters(), lr=LEARNING_RATE)

train_losses = np.zeros(NUM_EPOCH)
test_losses = np.zeros(NUM_EPOCH)

for epoch in range(NUM_EPOCH):
    rppg_model.train()
    t0 = datetime.now()
    train_loss = []
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
    test_loss = []
    for inputs, targets in test_loader:
        outputs = rppg_model(inputs)
        loss = criterion(outputs, targets.float())
        test_loss.append(loss.item())

    test_losses[epoch] = np.mean(test_loss)

    dt = datetime.now() - t0
    print(
        f"Epoch: {epoch + 1} / {NUM_EPOCH}, Train Loss: {train_losses[epoch]:.4f}, Test Loss: {test_losses[epoch]:.4f}, Duration: {dt}")


plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()
