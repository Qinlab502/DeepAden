# ================================================================
# trainer.py — Supervised Contrastive Learning
# ================================================================
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from dataset import ContrastiveDataset
from model import ContrastiveModel
from loss import ContrastiveLoss
from utils import evaluate_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


# ---------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss improved → {val_loss:.6f}. Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ---------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------
def train_one_fold(args, fold_idx, train_idx, val_idx, dataset, device):
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"\n🧩 Fold {fold_idx+1}/{args.k_folds} — Train={len(train_idx)} | Val={len(val_idx)}")
    model = ContrastiveModel(
        protein_dim=args.protein_dim,
        molecule_dim=args.molecule_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout
    ).to(device)
    criterion = ContrastiveLoss(args.temperature, args.lambda_contrastive)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"fold_{fold_idx+1}_best_{timestamp}.pth")
    es = EarlyStopping(patience=args.patience, verbose=False, delta=args.early_stopping_delta, path=save_path)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(args.epochs):
        # === Train ===
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}/{args.epochs}", leave=False):
            p = batch["protein_feature"].to(device)
            m = batch["molecule_feature"].to(device)
            labels = batch["label_index"].to(device).long()
            optimizer.zero_grad()
            p_proj, m_proj = model(p, m)
            loss = criterion(p_proj, m_proj, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Evaluate training set accuracy ===
        model.eval()
        with torch.no_grad():
            train_acc = evaluate_model(model, train_loader, mol_feature_dir=args.molecule_feature_dir, device=device)["accuracy"]
        train_accs.append(train_acc)

        # === Validate ===
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                p = batch["protein_feature"].to(device)
                m = batch["molecule_feature"].to(device)
                labels = batch["label_index"].to(device).long()
                p_proj, m_proj = model(p, m)
                loss = criterion(p_proj, m_proj, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = evaluate_model(model, val_loader, mol_feature_dir=args.molecule_feature_dir, device=device)["accuracy"]
        val_accs.append(val_acc)

        print(f"Fold {fold_idx+1} Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f} | Train Acc={train_acc*100:.2f}% | "
              f"Val Loss={avg_val_loss:.4f} | Val Acc={val_acc*100:.2f}%")

        scheduler.step()
        es(avg_val_loss, model)
        if es.early_stop:
            print(f"⛔ Early stopping triggered at epoch {epoch+1} (fold {fold_idx+1})")
            break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    return model, np.array(train_losses), np.array(val_losses), np.array(train_accs), np.array(val_accs)


# ---------------------------------------------------------------
# Plot aggregated metrics (Train/Val Loss + Accuracy)
# ---------------------------------------------------------------
def plot_kfold_metrics(fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs, save_dir):
    def pad_and_stack(list_arrays):
        max_len = max(len(a) for a in list_arrays)
        padded = []
        for a in list_arrays:
            if len(a) < max_len:
                pad_vals = np.ones(max_len - len(a)) * a[-1]
                a = np.concatenate([a, pad_vals])
            padded.append(a)
        return np.stack(padded)

    tr, vl, tr_acc, vl_acc = (
        pad_and_stack(fold_train_losses),
        pad_and_stack(fold_val_losses),
        pad_and_stack(fold_train_accs),
        pad_and_stack(fold_val_accs),
    )

    mtr, mvl, mtr_acc, mvl_acc = tr.mean(0), vl.mean(0), tr_acc.mean(0), vl_acc.mean(0)
    str_, svl, str_acc, svl_acc = tr.std(0), vl.std(0), tr_acc.std(0), vl_acc.std(0)
    epochs = np.arange(1, len(mtr) + 1)

    plt.figure(figsize=(9, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    # Loss curves
    ax1.plot(epochs, mtr, color="tab:orange", label="Train Loss")
    ax1.fill_between(epochs, mtr - str_, mtr + str_, alpha=0.2, color="tab:orange")
    ax1.plot(epochs, mvl, color="tab:red", label="Val Loss")
    ax1.fill_between(epochs, mvl - svl, mvl + svl, alpha=0.2, color="tab:red")
    # Accuracy curves
    ax2.plot(epochs, mtr_acc * 100, color="tab:blue", label="Train Acc (%)", linewidth=2)
    ax2.fill_between(epochs, (mtr_acc - str_acc) * 100, (mtr_acc + str_acc) * 100, alpha=0.15, color="tab:blue")
    ax2.plot(epochs, mvl_acc * 100, color="tab:green", label="Val Acc (%)", linewidth=2)
    ax2.fill_between(epochs, (mvl_acc - svl_acc) * 100, (mvl_acc + svl_acc) * 100, alpha=0.15, color="tab:green")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy (%)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("K-Fold Mean ± Std: Loss & Accuracy")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "kfold_train_val_metrics_joint.png")
    plt.savefig(out)
    plt.close()
    print(f"📊 Saved → {out}")
    

# ---------------------------------------------------------------
# 🚀 Train on full dataset + Early Stopping (monitor Train Loss)
# ---------------------------------------------------------------
def train_final_full_model(args, dataset, device):
    print("\n🚀 Training final model on ALL data with EarlyStopping (Train Loss monitor)...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ContrastiveModel(
        protein_dim=args.protein_dim,
        molecule_dim=args.molecule_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout
    ).to(device)
    criterion = ContrastiveLoss(args.temperature, args.lambda_contrastive)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"final_best_{timestamp}.pth")
    es = EarlyStopping(patience=args.patience, verbose=False, delta=args.early_stopping_delta, path=save_path)

    train_losses, train_accs = [], []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Final Model Epoch {epoch+1}/{args.epochs}", leave=False):
            p = batch["protein_feature"].to(device)
            m = batch["molecule_feature"].to(device)
            labels = batch["label_index"].to(device).long()
            optimizer.zero_grad()
            p_proj, m_proj = model(p, m)
            loss = criterion(p_proj, m_proj, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            metrics = evaluate_model(model, loader, mol_feature_dir=args.molecule_feature_dir, device=device)
        acc = metrics["accuracy"]
        train_accs.append(acc)
        print(f"[FINAL] Epoch {epoch+1}: Loss={avg_loss:.4f} | Train Acc={acc*100:.2f}%")

        scheduler.step()
        es(avg_loss, model)
        if es.early_stop:
            print(f"⛔ Early stopping triggered at epoch {epoch+1} (final full training)")
            break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    # --- Plot final metrics ---
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(epochs, train_losses, color='tab:red', label='Train Loss', linewidth=2)
    ax2.plot(epochs, np.array(train_accs) * 100, color='tab:blue', label='Train Accuracy (%)', linewidth=2)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color='tab:red')
    ax2.set_ylabel("Accuracy (%)", color='tab:blue')
    ax1.set_title("Final Model — Train Loss & Train Accuracy (Dual-Axis)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')
    out_plot = os.path.join(args.save_dir, "final_model_loss_acc_curve.png")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"📈 Final dual-axis training curve saved → {out_plot}")

    final_w = os.path.join(args.save_dir, "final_full_model.pth")
    torch.save(model.state_dict(), final_w)
    print(f"✅ Final model saved at: {final_w}")


# ---------------------------------------------------------------
# K-Fold wrapper
# ---------------------------------------------------------------
def cross_validate_contrastive_supervised(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"🧠 Using device: {device}")

    full_dataset = ContrastiveDataset(args.data_file, args.protein_feature_dir, args.molecule_feature_dir)
    all_labels = full_dataset.df["label"].map(full_dataset.molecule_to_index).values
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs, final_val_accs = [], [], [], [], []
    
    dummy_X = np.zeros(len(all_labels))
    
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(dummy_X, all_labels)):
        model, tr, vl, tr_acc, val_acc = train_one_fold(args, fold_idx, tr_idx, val_idx, full_dataset, device)
        fold_train_losses.append(tr)
        fold_val_losses.append(vl)
        fold_train_accs.append(tr_acc)
        fold_val_accs.append(val_acc)
        if len(val_acc) > 0:
            final_val_accs.append(val_acc[-1])

    print("\n📈 Cross-Validation Summary:")
    for i, acc in enumerate(final_val_accs):
        print(f"Fold {i+1}: Final Val Acc = {acc*100:.2f}%")
    print(f"Mean = {np.mean(final_val_accs)*100:.2f} ± {np.std(final_val_accs)*100:.2f}%")
    plot_kfold_metrics(fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs, args.save_dir)

    # 🚀 Train on full dataset
    train_final_full_model(args, full_dataset, device)


# ---------------------------------------------------------------
# Args
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Supervised Contrastive Trainer (K-fold + Dual-axis full training)")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--protein_feature_dir", type=str, required=True)
    parser.add_argument("--molecule_feature_dir", type=str, required=True)

    parser.add_argument("--protein_dim", type=int, default=1280)
    parser.add_argument("--molecule_dim", type=int, default=768)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stopping_delta", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="contrastive_kfold_fulltrain_jointacc")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cross_validate_contrastive_supervised(args)


if __name__ == "__main__":
    main()
