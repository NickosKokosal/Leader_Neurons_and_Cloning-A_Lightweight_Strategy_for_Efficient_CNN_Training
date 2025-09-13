import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Επιλογή συσκευής
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Χρησιμοποιούμε συσκευή: {device}")

# Μετασχηματισμοί για CIFAR-100
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Bottleneck block
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

# CNN Αρχιτεκτονική με Bottleneck
class LeaderCNN(nn.Module):
    def __init__(self):
        super(LeaderCNN, self).__init__()
        self.layer1 = Bottleneck(3, 128, 256)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = Bottleneck(256, 128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = Bottleneck(128, 32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.to(device)
        dummy_input = torch.zeros(1, 3, 32, 32).to(device)
        x = self.forward_features(dummy_input)
        linear_input_size = x.view(1, -1).shape[1]
        self.to("cpu")

        self.fc1 = nn.Linear(linear_input_size, 1024)
        self.fc2 = nn.Linear(1024, 100)

    def forward_features(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def elect_roles(weight_tensor, leaders_k, independents_k):
    total = weight_tensor.shape[0]
    all_indices = np.arange(total)
    leaders = np.random.choice(all_indices, size=leaders_k, replace=False)
    remaining = np.setdiff1d(all_indices, leaders)
    independents = np.random.choice(remaining, size=independents_k, replace=False)
    return leaders, independents

def train(model, train_loader, criterion, optimizer, epochs, leaders_k, independents_k, clone=True):
    model.train()
    loss_history = []
    log = []

    for epoch in range(epochs):
        epoch_loss = 0
        total_updates = 0
        start_time = time.time()

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()

            updates = 0
            if leaders_k > 0 or independents_k > 0:
                with torch.no_grad():
                    leaders, independents = elect_roles(model.layer1.conv1.weight.data, leaders_k, independents_k)
                    for i in range(model.layer1.conv1.weight.shape[0]):
                        if i not in leaders and i not in independents:
                            model.layer1.conv1.weight.grad[i] = torch.zeros_like(model.layer1.conv1.weight.grad[i])
                            model.layer1.conv1.bias.grad[i] = torch.zeros_like(model.layer1.conv1.bias.grad[i])
                        else:
                            updates += 1
            else:
                updates = model.layer1.conv1.weight.shape[0]

            total_updates += updates
            optimizer.step()

            if clone and (leaders_k > 0):
                with torch.no_grad():
                    leader_id = leaders[0]
                    for i in range(model.layer1.conv1.weight.shape[0]):
                        if i not in leaders and i not in independents:
                            model.layer1.conv1.weight[i] = model.layer1.conv1.weight[leader_id].clone()
                            model.layer1.conv1.bias[i] = model.layer1.conv1.bias[leader_id].clone()

            epoch_loss += loss.item()

        duration = time.time() - start_time
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Updates: {total_updates}, Time: {duration:.2f}s")
        loss_history.append(epoch_loss)
        log.append({"Epoch": epoch+1, "Loss": epoch_loss, "Updates": total_updates, "Time": duration})

    return loss_history, pd.DataFrame(log)

def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    accuracy = correct / total
    print(f"\nAccuracy στο test set: {accuracy:.2f}")
    return accuracy

scenarios = [
    {"desc": "(1)baseline training", "leaders": 0, "independents": 0, "clone": False},
    {"desc": "(2)leaders only", "leaders": 40 , "independents": 0, "clone": True},
    {"desc": "(3)leaders + independents", "leaders": 40, "independents": 10, "clone": True},  # 50
    {"desc": "(4)leaders + independents", "leaders": 35, "independents": 20, "clone": True},  # 55
    {"desc": "(5)leaders + independents", "leaders": 50, "independents": 25, "clone": True},  # 75
    {"desc": "(6)leaders + independents", "leaders": 60, "independents": 30, "clone": True},  # 90
]

all_histories = []

for config in scenarios:
    print(f"\n--- Τρέχουμε: {config['desc']} ---")
    model = LeaderCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    loss_history, log_df = train(
        model,
        train_loader,
        criterion,
        optimizer,
        epochs=25,
        leaders_k=config["leaders"],
        independents_k=config["independents"],
        clone=config["clone"]
    )

    accuracy = evaluate(model, test_loader)
    all_histories.append((config["desc"], loss_history))

for desc, losses in all_histories:
    plt.plot(losses, label=desc)

plt.title("Loss ανά Epoch για κάθε Σενάριο")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
