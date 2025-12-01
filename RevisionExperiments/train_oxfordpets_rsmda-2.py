import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset

from torchvision import datasets, transforms, models


# ---------------------- RSMDA for arbitrary resolution ---------------------- #

def rsmda_slice_mix(img_src, img_tgt, lam, mode="row"):
    """
    RSMDA: slice-based mixing between two images.

    img_src, img_tgt: tensors [C, H, W] in [0,1]
    lam: mixing parameter in [0,1]
    mode: 'row' (horizontal slices) or 'col' (vertical slices)

    Slice height S is drawn from [1, H/2] (or width for mode='col'),
    then N_mix = floor(lam * floor(H/S)) slices are replaced.
    """
    assert img_src.shape == img_tgt.shape, "Source and target must have same shape"
    C, H, W = img_src.shape

    if mode == "row":
        length = H
    else:
        length = W

    if length < 2:
        return img_src  # degenerate case

    # slice size S in [1, floor(length/2)]
    max_slice = max(1, length // 2)
    S = random.randint(1, max_slice)

    n_total = length // S
    if n_total == 0:
        return img_src

    n_mix = int(lam * n_total)
    if n_mix <= 0:
        return img_src

    # random starting offset in first half
    rand_pos = random.randint(0, max(0, length // 2))

    out = img_src.clone()

    # replace slices; use every second slice (0,2,4,...) to mimic your original code
    for i in range(0, n_mix, 2):
        start = rand_pos + i * S
        end = start + S

        if start >= length:
            break
        end = min(end, length)

        if mode == "row":
            out[:, start:end, :] = img_tgt[:, start:end, :]
        else:
            out[:, :, start:end] = img_tgt[:, :, start:end]

    return out



def rsmda_batch(inputs, targets, beta=1.0, mode="row", device="cuda", p=0.5):
    """
    Apply RSMDA to a *subset* of images in the batch, selected with probability p.

    inputs:  [B, C, H, W]
    targets: [B]
    beta:    Beta distribution parameter for lam
    p:       probability of applying RSMDA to a given image

    Returns:
        mixed   : [B, C, H, W]  (some rows augmented, some unchanged)
        target_a: [B]           (original labels)
        target_b: [B]           (permuted labels)
        lam_vec : [B] float     (lam for augmented samples, 1.0 for clean ones)
    """
    batch_size = inputs.size(0)
    device = inputs.device

    # partner indices
    index = torch.randperm(batch_size, device=device)

    inputs_a = inputs
    inputs_b = inputs[index]
    target_a = targets
    target_b = targets[index]

    mixed = inputs.clone()
    lam_vec = torch.ones(batch_size, device=device)  # default: no mix (lam = 1)

    # global lam draw (same lam for all augmented samples in this batch)
    if beta > 0:
        lam = float(np.random.beta(beta, beta))
    else:
        lam = 1.0

    for i in range(batch_size):
        if random.random() < p:
            # apply RSMDA to this sample
            mixed[i] = rsmda_slice_mix(inputs_a[i], inputs_b[i], lam, mode=mode)
            lam_vec[i] = lam   # mixed between target_a[i] and target_b[i]
        else:
            # no augmentation for this sample: keep original input & hard label
            mixed[i] = inputs_a[i]
            lam_vec[i] = 1.0

    return mixed, target_a, target_b, lam_vec


# -------------------------- Training / evaluation -------------------------- #

def train_one_epoch(model, loader, criterion, optimizer, device, use_rsmda=False, beta=1.0):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if use_rsmda:
            mixed_inputs, target_a, target_b, lam = rsmda_batch(
                inputs, targets, beta=beta, mode="row", device=device
            )
            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, target_a) + (1.0 - lam) * criterion(outputs, target_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # For accuracy, compare predictions to original targets (as in mixup practice)
        _, preds = outputs.max(1)
        running_correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * running_correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            running_correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * running_correct / total
    return epoch_loss, epoch_acc


# ------------------------------- Main script ------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet training with RSMDA")
    parser.add_argument("--data-dir", default="./data", type=str,
                        help="root directory for Oxford-IIIT Pet")
    parser.add_argument("--arch", default="resnet50",
                        choices=["resnet50", "mobilenet_v2"], help="model architecture")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--beta", default=1.0, type=float,
                        help="Beta distribution parameter for RSMDA (lam)")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use-rsmda", default=0, type=int)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------- Data / Dataloaders ------------------------- #
    print("==> Preparing Oxford-IIIT Pet dataset at 224x224")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.OxfordIIITPet(
        root=args.data_dir,
        split="trainval",
        download=True,
        transform=train_transform,
        target_types="category",
    )

    test_dataset = datasets.OxfordIIITPet(
        root=args.data_dir,
        split="test",
        download=True,
        transform=test_transform,
        target_types="category",
    )

    num_classes = len(train_dataset.classes)



    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------ Model -------------------------------- #
    print(f"==> Creating model {args.arch} (num_classes={num_classes})")

    if args.arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError("Unknown architecture")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --------------------------- Training loop --------------------------- #
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_rsmda=args.use_rsmda, beta=args.beta
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("checkpoints", exist_ok=True)
            save_path = os.path.join(
                "checkpoints",
                f"oxfordpets_{args.arch}_{'rsmda' if args.use_rsmda else 'baseline'}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"  [*] New best test acc: {best_acc:.2f}%  --> saved to {save_path}")

    print(f"\nBest test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
