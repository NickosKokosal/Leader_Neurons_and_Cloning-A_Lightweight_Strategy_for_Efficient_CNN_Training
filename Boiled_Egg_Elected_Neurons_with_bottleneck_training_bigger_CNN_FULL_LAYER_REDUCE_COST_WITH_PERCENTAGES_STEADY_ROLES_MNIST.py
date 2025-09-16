# MNIST + RoleAwareConv2d (Per-layer %) + Frozen roles per epoch
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

# Προαιρετικό: βοηθάει το cuDNN να βρει γρήγορους kernels όταν τα σχήματα είναι σταθερά
torch.backends.cudnn.benchmark = True

# -----------------------------
# ΣΥΣΚΕΥΗ
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# RoleAwareConv2d
#   - Υπολογίζει ΜΟΝΟ τους active (Leaders+Independents)
#   - Οι Clones ΔΕΝ κάνουν conv: παίρνουν feature maps από Leaders
#   - Αν δεν δοθούν ρόλοι -> λειτουργεί σαν κανονικό Conv2d
# -----------------------------
class RoleAwareConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias   = nn.Parameter(torch.empty(out_ch)) if bias else None
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.stride, self.padding = stride, padding
        self.dilation, self.groups = dilation, groups
        self._out_ch = out_ch

        # buffers για ρόλους (default: όλα active, κανένα clone)
        self.register_buffer('active_idx', torch.arange(out_ch))
        self.register_buffer('clone_idx',  torch.tensor([], dtype=torch.long))
        self.register_buffer('clone_src',  torch.tensor([], dtype=torch.long))

    @torch.no_grad()
    def set_roles(self, leaders, independents, clones, clone_sources):
        """Ορίζει ρόλους για το επόμενο forward."""
        dev = self.weight.device
        to_t = lambda x: torch.as_tensor(x, dtype=torch.long, device=dev)

        L = to_t(leaders) if len(leaders) else torch.tensor([], dtype=torch.long, device=dev)
        I = to_t(independents) if len(independents) else torch.tensor([], dtype=torch.long, device=dev)
        C = to_t(clones) if len(clones) else torch.tensor([], dtype=torch.long, device=dev)
        S = to_t(clone_sources) if len(clone_sources) else torch.tensor([], dtype=torch.long, device=dev)

        # Αν δεν υπάρχουν Leaders, δεν γίνεται cloning
        if L.numel() == 0:
            C = torch.tensor([], dtype=torch.long, device=dev)
            S = torch.tensor([], dtype=torch.long, device=dev)

        # Ασφάλεια: clones και sources ίδιο μήκος
        if C.numel() == 0 or S.numel() == 0 or C.numel() != S.numel():
            C = torch.tensor([], dtype=torch.long, device=dev)
            S = torch.tensor([], dtype=torch.long, device=dev)

        # Active = Leaders ∪ Independents (fallback: όλα active)
        A = torch.cat([L, I], dim=0)
        if A.numel() == 0:
            A = torch.arange(self._out_ch, device=dev, dtype=torch.long)

        self.active_idx = A
        self.clone_idx  = C
        self.clone_src  = S

    def forward(self, x):
        # Baseline μονοπάτι: όλα active, κανένα clone
        if self.active_idx.numel() == self._out_ch and self.clone_idx.numel() == 0:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Conv ΜΟΝΟ στα active out-channels
        W = self.weight.index_select(0, self.active_idx)
        b = self.bias.index_select(0, self.active_idx) if self.bias is not None else None
        y_active = F.conv2d(x, W, b, self.stride, self.padding, self.dilation, self.groups)  # [B, A, H, W]

        # Συναρμολόγηση πλήρους εξόδου
        B, A, H, W_ = y_active.shape
        y = torch.zeros(B, self._out_ch, H, W_, device=y_active.device, dtype=y_active.dtype)
        y.index_copy_(1, self.active_idx, y_active)

        # Clones: αντιγραφή feature maps από leaders (αν υπάρχει valid mapping)
        if self.clone_idx.numel() > 0 and self.clone_src.numel() == self.clone_idx.numel():
            y[:, self.clone_idx] = y[:, self.clone_src]
        return y

# -----------------------------
# Bottleneck block (MNIST μικρό)
# -----------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = RoleAwareConv2d(in_channels, mid_channels, k=1, bias=True)
        self.bn1   = nn.BatchNorm2d(mid_channels)

        self.conv2 = RoleAwareConv2d(mid_channels, mid_channels, k=3, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(mid_channels)

        self.conv3 = RoleAwareConv2d(mid_channels, out_channels, k=1, bias=True)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # Δεν αραιώνω τα shortcuts (κρατιούνται πλήρη για σταθερότητα των residuals)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

# -----------------------------
# Μοντέλο για MNIST
#   1×28×28 → MaxPool×3 → 10 κλάσεις
# -----------------------------
class LeaderCNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # Λίγα κανάλια (MNIST είναι «εύκολο»):
        self.layer1 = Bottleneck(1,  32,  64)   # in=1,   mid=32, out=64
        self.pool1  = nn.MaxPool2d(2,2)         # 28 -> 14
        self.layer2 = Bottleneck(64, 32,  64)   # in=64,  mid=32, out=64
        self.pool2  = nn.MaxPool2d(2,2)         # 14 -> 7
        self.layer3 = Bottleneck(64, 16,  32)   # in=64,  mid=16, out=32
        self.pool3  = nn.MaxPool2d(2,2)         #  7 -> 3
        self.dropout = nn.Dropout(0.3)

        # Υπολογισμός εισόδου για FC δυναμικά
        self.to(device)
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28, device=device)
            x = self.forward_features(dummy)    # αναμένουμε [1, 32, 3, 3] -> 288
            in_fc = x.view(1, -1).shape[1]
        self.train()
        self.to("cpu")

        self.fc1 = nn.Linear(in_fc, 128)
        self.fc2 = nn.Linear(128, 10)

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

# -----------------------------
# Εκλογή ρόλων από ποσοστά
# -----------------------------
def elect_roles_by_percent(out_channels: int, p_leaders: float, p_indep: float):
    """Επιστρέφει (leaders, independents, clones) ως numpy arrays, με clamp & κανονικοποίηση ποσοστών."""
    pL = max(0.0, min(1.0, p_leaders or 0.0))
    pI = max(0.0, min(1.0, p_indep   or 0.0))
    if pL + pI > 1.0:  # κανονικοποίηση για ασφάλεια
        s = 1.0 / (pL + pI)
        pL *= s; pI *= s

    kL = int(round(pL * out_channels))
    kI = int(round(pI * out_channels))
    kL = min(kL, out_channels)
    kI = min(kI, out_channels - kL)

    all_idx = np.arange(out_channels)
    leaders = np.random.choice(all_idx, size=kL, replace=False) if kL > 0 else np.array([], dtype=int)
    remaining = np.setdiff1d(all_idx, leaders)
    independents = (np.random.choice(remaining, size=kI, replace=False)
                    if kI > 0 else np.array([], dtype=int))
    used = np.union1d(leaders, independents)
    clones = np.setdiff1d(all_idx, used)
    return leaders, independents, clones

# -----------------------------
# ΕΚΠΑΙΔΕΥΣΗ (παγωμένοι ρόλοι ανά epoch)
# -----------------------------
def train(model, train_loader, criterion, optimizer,
          epochs=10,
          clone=True,
          per_layer_percent: dict | None = None,  # {"layerX.convY": (p_leaders, p_indep), ...}
          cloning_mode: str = "first",            # "first" ή "random"
          verbose: bool = False, print_every_batches: int = 200):

    model.train()
    loss_history, log = [], []

    # μαζεύω τα RoleAwareConv2d layers με τα ονόματά τους
    def list_role_convs(m: LeaderCNN_MNIST):
        return [
            ("layer1.conv1", m.layer1.conv1),
            ("layer1.conv2", m.layer1.conv2),
            ("layer1.conv3", m.layer1.conv3),
            ("layer2.conv1", m.layer2.conv1),
            ("layer2.conv2", m.layer2.conv2),
            ("layer2.conv3", m.layer2.conv3),
            ("layer3.conv1", m.layer3.conv1),
            ("layer3.conv2", m.layer3.conv2),
            ("layer3.conv3", m.layer3.conv3),
        ]
    target = list_role_convs(model)

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_updates = 0
        t0 = time.time()

        # 1) ΠΑΓΩΝΟΥΜΕ ρόλους για ΟΛΟ το epoch (μειώνει το overhead)
        roles_per_layer = []  # (lname, conv, L, I, C, src)
        with torch.no_grad():
            is_baseline = (per_layer_percent is None) or (len(per_layer_percent) == 0) or (not clone)
            for lname, conv in target:
                if is_baseline:
                    roles_per_layer.append((lname, conv,
                                            np.array([], int), np.array([], int),
                                            np.array([], int), np.array([], int)))
                    continue

                pL, pI = per_layer_percent.get(lname, (0.0, 0.0))
                out_ch = conv.weight.shape[0]
                L, I, C = elect_roles_by_percent(out_ch, pL, pI)

                if C.size > 0 and L.size > 0:
                    if cloning_mode == "first":
                        src = np.full_like(C, L[0])   # όλοι οι clones παίρνουν από τον 1ο leader
                    else:  # "random"
                        src = np.random.choice(L, size=len(C), replace=True)
                else:
                    C = np.array([], dtype=int)
                    src = np.array([], dtype=int)

                roles_per_layer.append((lname, conv, L, I, C, src))

        # 2) BATCH LOOP (εφαρμόζω τους ΙΔΙΟΥΣ ρόλους σε κάθε batch)
        for b_idx, (X, y) in enumerate(train_loader, start=1):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                batch_updates = 0
                for lname, conv, L, I, C, src in roles_per_layer:
                    conv.set_roles(L, I, C, src)
                    if (per_layer_percent is None) or (len(per_layer_percent) == 0) or (not clone):
                        batch_updates += conv.weight.shape[0]       # baseline: όλα active
                    else:
                        batch_updates += (len(L) + len(I))           # ενεργά: Leaders+Indep
                total_updates += batch_updates

            optimizer.zero_grad(set_to_none=True)
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if verbose and (b_idx % print_every_batches == 0):
                print(f"[Epoch {epoch+1} | Batch {b_idx}] loss={loss.item():.4f} | active-updates={batch_updates}")

        dt = time.time() - t0
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Updates: {total_updates}, Time: {dt:.2f}s")
        loss_history.append(epoch_loss)
        log.append({"Epoch": epoch+1, "Loss": epoch_loss, "Updates": total_updates, "Time": dt})

    return loss_history, pd.DataFrame(log)

# -----------------------------
# ΑΞΙΟΛΟΓΗΣΗ
# -----------------------------
def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"\nAccuracy στο test set: {acc:.4f}")
    return acc

# -----------------------------
# MAIN (Windows-safe)
# -----------------------------
def main():
    print(f"Χρησιμοποιούμε συσκευή: {device}")

    # Transforms για MNIST (mean/std από το dataset)
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),   # ελαφρύ augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST datasets
    train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    # DataLoaders (σε Windows, αν δω θέμα, βάζω num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # --------- per-layer ποσοστά (leaders%, independents%) ---------
    # Για MNIST είναι λίγα κανάλια → κρατάμε πιο «ήπια» ποσοστά στα μικρά layers
    mild_per_layer = {
        # layer1 (mid=32, out=64)
        "layer1.conv1": (0.40, 0.10),
        "layer1.conv2": (0.40, 0.10),
        "layer1.conv3": (0.35, 0.10),
        # layer2 (mid=32, out=64)
        "layer2.conv1": (0.40, 0.10),
        "layer2.conv2": (0.40, 0.10),
        "layer2.conv3": (0.35, 0.10),
        # layer3 (mid=16, out=32) — πολύ λίγα κανάλια, κρατάμε περισσότερους leaders
        "layer3.conv1": (0.50, 0.10),
        "layer3.conv2": (0.50, 0.10),
        "layer3.conv3": (0.45, 0.10),
    }

    aggressive_per_layer = {
        "layer1.conv1": (0.30, 0.10),
        "layer1.conv2": (0.30, 0.10),
        "layer1.conv3": (0.25, 0.10),
        "layer2.conv1": (0.30, 0.10),
        "layer2.conv2": (0.30, 0.10),
        "layer2.conv3": (0.25, 0.10),
        "layer3.conv1": (0.40, 0.10),
        "layer3.conv2": (0.40, 0.10),
        "layer3.conv3": (0.35, 0.10),
    }

    # ΣΕΝΑΡΙΑ: baseline / mild / aggressive
    scenarios = [
        {"desc": "(1)baseline",   "per_layer_p": None,               "clone": False},
        {"desc": "(2)mild %",     "per_layer_p": mild_per_layer,     "clone": True},
        {"desc": "(3)aggressive %","per_layer_p": aggressive_per_layer,"clone": True},
    ]

    all_histories = []
    for cfg in scenarios:
        print(f"\n--- Τρέχουμε: {cfg['desc']} ---")
        model = LeaderCNN_MNIST().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        loss_history, _ = train(
            model, train_loader, criterion, optimizer,
            epochs=10,                      # MNIST: λίγα epochs για demo (αύξησέ τα αν θέλεις)
            clone=cfg["clone"],
            per_layer_percent=cfg["per_layer_p"],
            cloning_mode="first",          # λιγότερο overhead
            verbose=False
        )
        acc = evaluate(model, test_loader)
        all_histories.append((cfg["desc"], loss_history))

    # Plot καμπύλες loss
    for desc, losses in all_histories:
        plt.plot(losses, label=desc)
    plt.title("MNIST — Loss ανά Epoch (Baseline vs Mild/Aggr per-layer %)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Windows-safe entry point
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
# ======================================================================================================
