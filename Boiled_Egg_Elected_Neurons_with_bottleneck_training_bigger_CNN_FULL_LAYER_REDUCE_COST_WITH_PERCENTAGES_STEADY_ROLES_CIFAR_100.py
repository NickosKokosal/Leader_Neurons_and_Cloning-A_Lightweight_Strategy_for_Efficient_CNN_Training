# ============================ FULL SCRIPT (Per-layer % + RoleAwareConv2d + Frozen Roles/epoch) ============================
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
import random

# (Προαιρετικά για αναπαραγωγιμότητα)
# SEED = 42
# random.seed(SEED); np.random.seed(SEED)
# torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Βοηθάει την GPU να βρει ταχύτερα kernels όταν τα σχήματα είναι σταθερά (εδώ: ίδιοι ρόλοι σε όλο το epoch)
torch.backends.cudnn.benchmark = True

# -----------------------------------------------
# ΡΥΘΜΙΣΗ ΣΥΣΚΕΥΗΣ
# -----------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------
# RoleAwareConv2d: Υπολογίζει ΜΟΝΟ τα active κανάλια (leaders+independents).
# Τα clones ΔΕΝ κάνουν conv — αντιγράφουν feature maps από leaders.
# -----------------------------------------------
class RoleAwareConv2d(nn.Module):
    """
    Conv2d που δέχεται ρόλους:
      - leaders, independents -> ACTIVE (υπολογίζονται με F.conv2d)
      - clones                -> ΔΕΝ υπολογίζονται· αντιγράφουν feature maps από leader sources
    Αν δεν οριστούν ρόλοι (default), λειτουργεί σαν κανονικό Conv2d (full-conv).
    """
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias   = nn.Parameter(torch.empty(out_ch)) if bias else None
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation
        self.groups   = groups
        self._out_ch  = out_ch

        # buffers για ρόλους (default: όλα active, κανείς clone)
        self.register_buffer('active_idx', torch.arange(out_ch))
        self.register_buffer('clone_idx',  torch.tensor([], dtype=torch.long))
        self.register_buffer('clone_src',  torch.tensor([], dtype=torch.long))

    @torch.no_grad()
    def set_roles(self, leaders, independents, clones, clone_sources):
        """
        Ορίζει ρόλους για ΤΟ ΕΠΟΜΕΝΟ forward.
        leaders/independents/clones: iterable indices (numpy ή torch).
        clone_sources: για κάθε clone -> από ποιο leader αντιγράφει.
        Αν δεν κληθεί, τρέχει full-conv (όλα active).
        """
        dev = self.weight.device
        to_t = lambda x: torch.as_tensor(x, dtype=torch.long, device=dev)

        # Μετατροπές σε tensors (αν είναι άδειες λίστες, γίνονται κενά tensors)
        L = to_t(leaders) if len(leaders) else torch.tensor([], dtype=torch.long, device=dev)
        I = to_t(independents) if len(independents) else torch.tensor([], dtype=torch.long, device=dev)
        C = to_t(clones) if len(clones) else torch.tensor([], dtype=torch.long, device=dev)
        S = to_t(clone_sources) if len(clone_sources) else torch.tensor([], dtype=torch.long, device=dev)

        # Αν δεν υπάρχουν leaders, δεν έχει νόημα cloning
        if L.numel() == 0:
            C = torch.tensor([], dtype=torch.long, device=dev)
            S = torch.tensor([], dtype=torch.long, device=dev)

        # Ασφάλεια: clones και sources πρέπει να είναι ισομήκη
        if C.numel() == 0 or S.numel() == 0 or C.numel() != S.numel():
            C = torch.tensor([], dtype=torch.long, device=dev)
            S = torch.tensor([], dtype=torch.long, device=dev)

        # Active = leaders ∪ independents. Αν δεν έχει active, κάνε όλα active (fallback/baseline).
        A = torch.cat([L, I], dim=0)
        if A.numel() == 0:
            A = torch.arange(self._out_ch, device=dev, dtype=torch.long)

        self.active_idx = A
        self.clone_idx  = C
        self.clone_src  = S

    def forward(self, x):
        # Αν όλα active και κανείς clone -> πλήρης conv (baseline)
        if self.active_idx.numel() == self._out_ch and self.clone_idx.numel() == 0:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # 1) Conv ΜΟΝΟ στα active out-channels
        W = self.weight.index_select(0, self.active_idx)                    # [A, in_ch, k, k]
        b = self.bias.index_select(0, self.active_idx) if self.bias is not None else None
        y_active = F.conv2d(x, W, b, self.stride, self.padding, self.dilation, self.groups)  # [B, A, H, W]

        # 2) Συναρμολόγηση πλήρους εξόδου [B, out_ch, H, W]
        B, A, H, W_ = y_active.shape
        y = torch.zeros(B, self._out_ch, H, W_, device=y_active.device, dtype=y_active.dtype)
        y.index_copy_(1, self.active_idx, y_active)

        # 3) Clones: μόνο αν έχουμε valid mapping
        if self.clone_idx.numel() > 0 and self.clone_src.numel() == self.clone_idx.numel():
            y[:, self.clone_idx] = y[:, self.clone_src]

        return y

# -----------------------------------------------
# Bottleneck block (με RoleAwareConv2d)
# -----------------------------------------------
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
            # Συνήθως δεν αραιώνω τα shortcuts.
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

# -----------------------------------------------
# Κύριο μοντέλο
# -----------------------------------------------
class LeaderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Bottleneck(3,   128, 256)
        self.pool1  = nn.MaxPool2d(2, 2)    # 32->16
        self.layer2 = Bottleneck(256, 128, 128)
        self.pool2  = nn.MaxPool2d(2, 2)    # 16->8
        self.layer3 = Bottleneck(128,  32,  64)
        self.pool3  = nn.MaxPool2d(2, 2)    #  8->4
        self.dropout = nn.Dropout(0.5)

        # Dynamic FC input
        self.to(device)
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32, device=device)
            x = self.forward_features(dummy)  # (1, 64, 4, 4)
            in_fc = x.view(1, -1).shape[1]    # 64*4*4 = 1024
        self.train()
        self.to("cpu")

        self.fc1 = nn.Linear(in_fc, 1024)
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

# -----------------------------------------------
# Εκλογή ρόλων από ποσοστά (ανά layer)
# -----------------------------------------------
def elect_roles_by_percent(out_channels: int, p_leaders: float, p_indep: float):
    """
    Δίνει ρόλους με βάση ποσοστά (ανά layer).
    Επιστρέφει numpy arrays: (leaders, independents, clones)
    """
    # clamp ποσοστών [0,1] και κανονικοποίηση να μην ξεπερνούν αθροιστικά το 1
    pL = max(0.0, min(1.0, p_leaders or 0.0))
    pI = max(0.0, min(1.0, p_indep   or 0.0))
    if pL + pI > 1.0:
        s = 1.0/(pL+pI)
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

# -----------------------------------------------
# ΕΚΠΑΙΔΕΥΣΗ (per-layer ποσοστά, ΠΑΓΩΜΕΝΟΙ ρόλοι ανά epoch → χαμηλότερο overhead)
# -----------------------------------------------
def train(model, train_loader, criterion, optimizer,
          epochs=25,
          clone=True,
          per_layer_percent: dict | None = None,  # π.χ. {"layer1.conv1": (0.30, 0.10), ...}
          cloning_mode: str = "first",            # "first" (default) ή "random" για mapping clone->leader
          weight_sync: bool = False,              # αν True: συγχρονίζει και ΤΑ ΒΑΡΗ clone=leader μετά το step (προαιρετικό)
          verbose: bool = False, print_every_batches: int = 200):
    """
    - Αν per_layer_percent είναι None ή κενό → baseline (full conv, χωρίς ρόλους).
    - Διαφορετικά, για ΚΑΘΕ layer παίρνουμε (p_leaders, p_indep).
    - Οι ρόλοι εκλέγονται ΜΙΑ ΦΟΡΑ στην αρχή κάθε epoch (ΠΑΓΩΜΕΝΟΙ ρόλοι) και εφαρμόζονται σε ΟΛΑ τα batches του epoch.
      Αυτό μειώνει σημαντικά το runtime overhead σε σχέση με εκλογή ανά batch.
    """
    model.train()
    loss_history, log = [], []

    # Συγκέντρωση των RoleAwareConv2d layers για να τους περνάω ρόλους
    def list_role_convs(m: LeaderCNN):
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

        # -------- 1) ΠΑΓΩΜΕΝΟΙ ΡΟΛΟΙ ΓΙΑ ΟΛΟ ΤΟ EPOCH --------
        roles_per_layer = []  # (lname, conv, L, I, C, src)
        with torch.no_grad():
            is_baseline = (per_layer_percent is None) or (len(per_layer_percent) == 0) or (not clone)
            for lname, conv in target:
                if is_baseline:
                    # baseline ή cloning off -> full conv
                    roles_per_layer.append((lname, conv,
                                            np.array([], int), np.array([], int),
                                            np.array([], int), np.array([], int)))
                    continue

                # ποσοστά leaders/independents (αν δεν υπάρχει στο dict -> (0,0) → full layer)
                pL, pI = per_layer_percent.get(lname, (0.0, 0.0))
                out_ch = conv.weight.shape[0]
                L, I, C = elect_roles_by_percent(out_ch, pL, pI)

                if C.size > 0 and L.size > 0:
                    if cloning_mode == "first":
                        src = np.full_like(C, L[0])   # όλοι οι clones παίρνουν από τον 1ο leader
                    elif cloning_mode == "random":
                        src = np.random.choice(L, size=len(C), replace=True)
                    else:
                        src = np.full_like(C, L[0])
                else:
                    C = np.array([], dtype=int)
                    src = np.array([], dtype=int)

                roles_per_layer.append((lname, conv, L, I, C, src))

        # -------- 2) BATCH LOOP --------
        for b_idx, (X, y) in enumerate(train_loader, start=1):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Εφαρμόζω τους ΙΔΙΟΥΣ ρόλους σε κάθε batch (παγωμένοι για το epoch)
            with torch.no_grad():
                batch_updates = 0
                for lname, conv, L, I, C, src in roles_per_layer:
                    conv.set_roles(L, I, C, src)
                    if (per_layer_percent is None) or (len(per_layer_percent) == 0) or (not clone):
                        batch_updates += conv.weight.shape[0]   # baseline: όλα active
                    else:
                        batch_updates += (len(L) + len(I))       # ενεργά = leaders + independents
                total_updates += batch_updates

            optimizer.zero_grad(set_to_none=True)
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # (προαιρετικό) Συγχρονισμός ΒΑΡΩΝ clones = leader μετά το step (ΔΕΝ απαιτείται για compute saving)
            if weight_sync and clone and (per_layer_percent is not None) and (len(per_layer_percent) > 0):
                with torch.no_grad():
                    for lname, conv, L, I, C, src in roles_per_layer:
                        if len(C) == 0 or len(L) == 0:
                            continue
                        c_idx = torch.as_tensor(C, dtype=torch.long, device=conv.weight.device)
                        s_idx = torch.as_tensor(src, dtype=torch.long, device=conv.weight.device)
                        conv.weight[c_idx] = conv.weight[s_idx].clone()
                        if conv.bias is not None:
                            conv.bias[c_idx] = conv.bias[s_idx].clone()

            if verbose and (b_idx % print_every_batches == 0):
                print(f"[Epoch {epoch+1} | Batch {b_idx}] loss={loss.item():.4f} | active-updates={batch_updates}")

            epoch_loss += loss.item()

        dt = time.time() - t0
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Updates: {total_updates}, Time: {dt:.2f}s")
        loss_history.append(epoch_loss)
        log.append({"Epoch": epoch+1, "Loss": epoch_loss, "Updates": total_updates, "Time": dt})

    return loss_history, pd.DataFrame(log)

# -----------------------------------------------
# ΑΞΙΟΛΟΓΗΣΗ
# -----------------------------------------------
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
    print(f"\nAccuracy στο test set: {acc:.3f}")
    return acc

# -----------------------------------------------
# MAIN: datasets/loaders, σενάρια, run, plots
# -----------------------------------------------
def main():
    print(f"Χρησιμοποιούμε συσκευή: {device}")

    # Μετασχηματισμοί: train με augmentations, test καθαρό
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    # CIFAR-100
    train_dataset = datasets.CIFAR100(root='./data', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    # Dataloaders (αν δω Windows multiprocessing θέμα, βάζω προσωρινά num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # --------- ΟΡΙΣΜΟΣ per-layer ποσοστών ---------
    # (leaders%, independents%) — clones% = 1 - (leaders% + independents%)
    mild_per_layer = {
        # layer1
        "layer1.conv1": (0.40, 0.10),  # 128ch -> ~38 L, ~13 I
        "layer1.conv2": (0.40, 0.10),  # 128ch
        "layer1.conv3": (0.45, 0.20),  # 256ch -> ~64 L, ~26 I
        # layer2
        "layer2.conv1": (0.40, 0.10),  # 128ch
        "layer2.conv2": (0.40, 0.10),  # 128ch
        "layer2.conv3": (0.40, 0.10),  # 128ch
        # layer3 (λίγα κανάλια → λίγο μεγαλύτερο pL για σταθερότητα)
        "layer3.conv1": (0.35, 0.10),  # 32ch  -> ~11 L, ~3 I
        "layer3.conv2": (0.35, 0.10),  # 32ch
        "layer3.conv3": (0.30, 0.10),  # 64ch  -> ~19 L, ~6 I
    }

    aggressive_per_layer = {
        "layer1.conv1": (0.20, 0.10),
        "layer1.conv2": (0.20, 0.10),
        "layer1.conv3": (0.18, 0.10),
        "layer2.conv1": (0.20, 0.10),
        "layer2.conv2": (0.20, 0.10),
        "layer2.conv3": (0.20, 0.10),
        "layer3.conv1": (0.25, 0.10),
        "layer3.conv2": (0.25, 0.10),
        "layer3.conv3": (0.22, 0.10),
    }

    # SCENARIOS
    scenarios = [
        # (1) Baseline: χωρίς ρόλους -> full conv
        {"desc": "(1)baseline training",      "per_layer_p": None,                "clone": False},
        # (2) Ήπιο per-layer
        {"desc": "(2)mild per-layer",         "per_layer_p": mild_per_layer,      "clone": True},
        # (3) Επιθετικό per-layer
        {"desc": "(3)aggressive per-layer",   "per_layer_p": aggressive_per_layer,"clone": True},
    ]

    all_histories = []
    for cfg in scenarios:
        print(f"\n--- Τρέχουμε: {cfg['desc']} ---")
        model = LeaderCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        loss_history, log_df = train(
            model, train_loader, criterion, optimizer,
            epochs=25,
            clone=cfg["clone"],
            per_layer_percent=cfg["per_layer_p"],   # <-- per-layer ποσοστά
            cloning_mode="first",                   # λιγότερο overhead από "random"
            weight_sync=False,
            verbose=False, print_every_batches=200
        )

        acc = evaluate(model, test_loader)
        all_histories.append((cfg["desc"], loss_history))

    # Καμπύλες loss
    for desc, losses in all_histories:
        plt.plot(losses, label=desc)
    plt.title("Loss ανά Epoch (Per-layer Leaders/Independents %, Frozen per-epoch roles)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Windows-safe entry point (απαραίτητο με num_workers>0)
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
# =====================================================================================================
