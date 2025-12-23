# =========================
# Imports & Utility Classes
# =========================
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange
import prettytable as pt
from Bio import SeqIO
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
from transformers import EsmTokenizer, EsmModel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from deepAden_model import MLP, GAT

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Configuration Class
# =========================
class Config:
    def __init__(self):
        self.Dataset_dir = os.path.abspath('.') + '/ABP_data'
        # Model parameters
        self.hidden_channels = 64
        self.out_channels = 3 
        self.heads = 32
        self.heads_intermediate = 16
        self.heads_final = 8
        self.dropratio = 0.5
        self.input_dim_esm = 1280
        self.input_dim_physio = 18
        self.fusion_dim = 512  # D

        # Training parameters
        self.str_lr = 0.001
        self.L2_weight = 0
        self.epoch = 700
        self.early_stop_epochs = 200
        self.saved_model_num = 1
        self.train = True
        self.model_path = os.path.join(self.Dataset_dir, 'model')
        self.batch_size = 8

        # GRAND-specific parameters
        self.dropnode_rate = 0.4
        self.dropedge_rate = 0.4
        self.tem = 0.5
        self.lam_initial = 0.1
        self.lam_final = 0.5
        self.lam_rampup_epochs = 300
        self.order = 1

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def print_config(self):
        """Print all configuration parameters."""
        for name, value in vars(self).items():
            print(f'{name} = {value}')


# =========================
# Utility Functions for Data Handling
# =========================

def load_annotations(dataset_dir, dataset_files):
    """
    Load sequence annotations for training and testing datasets.
    """
    trainset_anno = os.path.join(dataset_dir, dataset_files['train'])
    testset_anno = os.path.join(dataset_dir, dataset_files['test'])

    seqanno = {}
    train_list = []
    test_list = []

    with open(trainset_anno, 'r') as f:
        train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}

    with open(testset_anno,'r') as f:
        test_text = f.readlines()
        for i in range(0, len(test_text), 3):
            query_id = test_text[i].strip()[1:]
            query_seq = test_text[i + 1].strip()
            query_anno = test_text[i + 2].strip()
            test_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
    
    return train_list, test_list, seqanno

def tv_split(train_list, seed=1995):
    """
    Split dataset into training and validation sets.
    """
    random.seed(seed)
    random.shuffle(train_list)
    valid_size = int(len(train_list) * 0.1)
    valid_list = train_list[:valid_size]
    train_list = train_list[valid_size:]
    return train_list, valid_list

def StatisticsSampleNum(train_list, valid_list, test_list, seqanno):
    """
    Print statistics for dataset splits.
    """
    def sub(seqlist, seqanno):
        pos_num_all = 0
        res_num_all = 0
        for seqid in seqlist:
            anno = list(map(int, list(seqanno[seqid]['anno'])))
            pos_num = sum(anno)
            res_num = len(anno)
            pos_num_all += pos_num
            res_num_all += res_num
        neg_num_all = res_num_all - pos_num_all
        pnratio = pos_num_all / float(neg_num_all) if neg_num_all != 0 else 0
        return len(seqlist), res_num_all, pos_num_all, neg_num_all, pnratio

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset','NumSeq', 'NumRes', 'NumPos', 'NumNeg', 'PNratio']
    tb.float_format = '0.3'

    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list + valid_list, seqanno)
    tb.add_row(['train+valid', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list, seqanno)
    tb.add_row(['train', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(valid_list, seqanno)
    tb.add_row(['valid', seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(test_list, seqanno)
    tb.add_row(['test', seq_num, res_num, pos_num, neg_num, pnratio])
    print(tb)

def create_pyg_data_train(
    esm_features: Dict[str, np.ndarray],
    physio_features: Dict[str, np.ndarray],
    edge_indices: Dict[str, np.ndarray],
    seqanno: Dict[str, Dict[str, str]],
    dataset_splits: Dict[str, List[str]],
    is_labeled: Dict[str, bool]
):
    """
    Create PyTorch Geometric Data objects for each split.
    Handles both labeled and unlabeled data.
    """
    processed_data = {}
    for dataset_type, seq_ids in dataset_splits.items():
        data_list = []
        for seq_id in tqdm(seq_ids, desc=f"Processing {dataset_type} data"):
            esm_emb = esm_features.get(seq_id)
            physio_feat = physio_features.get(seq_id)
            edge_idx = edge_indices.get(seq_id)

            if esm_emb is None or physio_feat is None or edge_idx is None:
                print(f"Warning: {seq_id} is missing features or edges and will be skipped.")
                continue

            if esm_emb.shape[0] != physio_feat.shape[0]:
                print(f"Warning: Mismatch in number of residues for {seq_id}.")
                continue

            # Create tensors
            features_esm = torch.tensor(esm_emb, dtype=torch.float)
            features_physio = torch.tensor(physio_feat, dtype=torch.float)
            edge_index_tensor = torch.tensor(edge_idx, dtype=torch.long)

            if is_labeled.get(seq_id, False):
                anno = seqanno.get(seq_id, {}).get('anno')
                if anno is None:
                    print(f"Warning: {seq_id} is marked as labeled but has no annotations.")
                    continue
                labels = torch.tensor([int(x) for x in anno], dtype=torch.long)
                mask = labels != 0
            else:
                num_nodes = esm_emb.shape[0]
                labels = torch.full((num_nodes,), -1, dtype=torch.long)
                mask = torch.zeros(num_nodes, dtype=torch.bool)

            data = Data(
                x_esm=features_esm, 
                x_physio=features_physio, 
                edge_index=edge_index_tensor, 
                y=labels,
                mask=mask
            )
            data.num_nodes = features_esm.shape[0]
            data_list.append(data)

        processed_data[dataset_type] = data_list
        print(f"Processed {len(data_list)} items for {dataset_type}.")
    return processed_data

# =========================
# Loss, Augmentation & Regularization
# =========================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=2.0, reduction='mean', num_classes=3):
        """
        Focal loss for imbalanced classes.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_tensor = self.alpha * torch.ones_like(targets, dtype=torch.float32)
        at = alpha_tensor.gather(0, targets.data.view(-1))
        loss = at * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def consis(logits_augmented: torch.Tensor, logits_labeled: torch.Tensor, temp: float = 0.5):
    """
    Consistency regularization loss between augmented and original logits.
    """
    ps_labeled = F.softmax(logits_labeled, dim=1)
    ps_augmented = F.softmax(logits_augmented, dim=1)
    avg_p = (ps_labeled + ps_augmented) / 2
    sharp_p = (torch.pow(avg_p, 1. / temp) 
               / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = F.mse_loss(ps_labeled, sharp_p, reduction='mean') + F.mse_loss(ps_augmented, sharp_p, reduction='mean')
    loss = loss / 2
    return loss

def propagate(feature, edge_index, order, device):
    """
    Simple feature propagation for node features.
    """
    x = feature
    y = feature.clone()
    for _ in range(order):
        row, col = edge_index
        agg = torch.zeros_like(x).to(device)
        agg.index_add_(0, row, x[col])
        y = y + agg
    return y / (order + 1)

def rand_prop(features, edge_index, label_mask, training, config):
    """
    Random propagation with DropNode and DropEdge for targeted nodes/edges.
    """
    n = features.shape[0]
    drop_rate_node = config.dropnode_rate
    drop_rate_edge = config.dropedge_rate

    if training:
        node_keep_prob = torch.where(
            label_mask,
            1.0 - drop_rate_node,
            torch.ones_like(label_mask, dtype=torch.float)
        )
        node_keep_mask = torch.bernoulli(node_keep_prob).bool()
        mask = node_keep_mask | (~label_mask)
        features = features * mask.unsqueeze(1)
    else:
        features = features * (
            torch.where(
                label_mask,
                1.0 - drop_rate_node,
                torch.ones_like(label_mask, dtype=torch.float)
            ).unsqueeze(1)
        )

    # DropEdge only for edges connected to labeled nodes
    if training and drop_rate_edge > 0.0:
        src, dst = edge_index
        labeled_edges_mask = label_mask[src] | label_mask[dst]
        edge_keep_prob = 1.0 - drop_rate_edge
        edge_keep_mask = torch.bernoulli(torch.full((edge_index.size(1),), edge_keep_prob, device=edge_index.device)).bool()
        final_edge_mask = (~labeled_edges_mask) | (labeled_edges_mask & edge_keep_mask)
        edge_index = edge_index[:, final_edge_mask]
        if edge_index.size(1) == 0:
            edge_index = edge_index
    return propagate(features, edge_index, config.order, features.device), edge_index

def calculate_current_lam(epoch, config):
    """
    Calculate the current lambda for consistency loss ramp-up.
    """
    if epoch >= config.lam_rampup_epochs:
        return config.lam_final
    else:
        return config.lam_initial + (config.lam_final - config.lam_initial) * np.exp(-5 * (config.lam_rampup_epochs - epoch) / config.lam_rampup_epochs)

# =========================
# Evaluation & Training Functions
# =========================

def eval_metrics(probs, targets, num_classes):
    """
    Compute accuracy, precision, recall, and F1-score for predictions.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if probs.ndim == 1:
        pred = probs
    else:
        pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(targets, pred, labels=np.arange(num_classes))
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        targets, pred, average='macro', zero_division=1
    )
    accuracy = np.trace(cm) / np.sum(cm)

    return accuracy, precision, recall, f1_score

def print_results(valid_matrices=None, test_matrices=None):
    """
    Print evaluation results for validation and test sets.
    """
    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'Acc', 'Pre', 'Rec', 'F1']
    datasets = [('valid', valid_matrices), ('test', test_matrices)]
    for name, metrics in datasets:
        if metrics is not None:
            row = [name] + [f"{x:.3f}" for x in metrics]
            tb.add_row(row)
    print(tb)

def evaluate_loader(model, criterion, loader, config, device):
    """
    Evaluate the model on a loader and compute metrics.
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_aa = model.mlp_aa(batch.x_esm)
            x_pc = model.mlp_pc(batch.x_physio)
            x_labeled = x_aa + x_pc
            output = model(x_labeled, batch.edge_index)
            loss = criterion(output, batch.y)
            total_loss += loss.item()
            all_outputs.append(output)
            all_labels.append(batch.y)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = eval_metrics(all_outputs, all_labels, config.out_channels)
    avg_loss = total_loss / len(loader)
    return avg_loss, metrics

def train(
    model, 
    optimizer, 
    criterion, 
    consis, 
    config, 
    train_loader, 
    unlabeled_loader, 
    val_loader, 
    test_loader, 
    device, 
    scheduler
):
    """
    Training loop for semi-supervised learning.
    """
    t_total = time.time()
    bad_counter = 0
    f1_best = 0.0
    best_epoch = 0
    unlabeled_iter = iter(unlabeled_loader)

    for epoch in range(1, config.epoch + 1):
        model.train()
        total_supervised_loss = 0.0
        total_consis_loss = 0.0
        total_loss = 0.0
        current_lam = calculate_current_lam(epoch, config)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training | lam={current_lam:.3f}")
        for batch_idx, batch in enumerate(progress_bar):
            labeled_batch = batch.to(device)
            optimizer.zero_grad()

            # Supervised loss
            x_aa = model.mlp_aa(labeled_batch.x_esm)
            x_pc = model.mlp_pc(labeled_batch.x_physio)
            x_labeled = x_aa + x_pc
            logits_labeled = model(x_labeled, labeled_batch.edge_index)
            supervised_loss = criterion(logits_labeled, labeled_batch.y)

            # Unlabeled batch
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            unlabeled_batch = unlabeled_batch.to(device)

            with torch.no_grad():
                x_aa_u = model.mlp_aa(unlabeled_batch.x_esm)
                x_pc_u = model.mlp_pc(unlabeled_batch.x_physio)
                x_unlabeled = x_aa_u + x_pc_u

            labeled_mask_labeled = labeled_batch.mask
            labeled_mask_unlabeled = unlabeled_batch.mask

            x_aug_unlabeled, edge_aug_unlabeled = rand_prop(
                x_unlabeled, 
                unlabeled_batch.edge_index, 
                labeled_mask_unlabeled, 
                training=True, 
                config=config
            )
            x_aug_labeled, edge_aug_labeled = rand_prop(
                x_labeled, 
                labeled_batch.edge_index, 
                labeled_mask_labeled, 
                training=True, 
                config=config
            )

            logits_aug_unlabeled = model(x_aug_unlabeled, edge_aug_unlabeled)
            logits_unlabeled = model(x_unlabeled, unlabeled_batch.edge_index)
            logits_aug_labeled = model(x_aug_labeled, edge_aug_labeled)

            consistency_loss_unlabeled = consis(logits_aug_unlabeled, logits_unlabeled, temp=config.tem)
            consistency_loss_labeled = consis(logits_aug_labeled, logits_labeled, temp=config.tem)
            consistency_loss = (consistency_loss_unlabeled + consistency_loss_labeled) / 2

            # Total loss
            loss = supervised_loss + current_lam * consistency_loss
            loss.backward()
            optimizer.step()

            total_supervised_loss += supervised_loss.item()
            total_consis_loss += consistency_loss.item()
            total_loss += loss.item()

            progress_bar.set_postfix({
                'Supervised Loss': f"{supervised_loss.item():.4f}",
                'Consis Loss': f"{consistency_loss.item():.4f}",
                'Total Loss': f"{loss.item():.4f}"
            })

        # Validation
        loss_val, metrics_val = evaluate_loader(model, criterion, val_loader, config, device)
        acc_val, pre_val, rec_val, f1_val = metrics_val

        if scheduler is not None:
            scheduler.step(f1_val)

        avg_supervised_loss = total_supervised_loss / len(train_loader)
        avg_consis_loss = total_consis_loss / len(train_loader)
        total_loss_value = avg_supervised_loss + current_lam * avg_consis_loss
        print(f"|| Epoch {epoch}/{config.epoch} || "
              f"train_supervised_loss={avg_supervised_loss:.5f} "
              f"+ {current_lam:.3f} * train_consis_loss={avg_consis_loss:.5f} = total_loss={total_loss_value:.5f} | "
              f"val_acc={acc_val:.3f} pre={pre_val:.3f} rec={rec_val:.3f} F1={f1_val:.3f}")

        # Test
        loss_test, metrics_test = evaluate_loader(model, criterion, test_loader, config, device)
        acc_test, pre_test, rec_test, f1_test = metrics_test
        print(f"Test Result: acc={acc_test:.3f} pre={pre_test:.3f} rec={rec_test:.3f} F1={f1_test:.3f}")

        # Model checkpointing and early stopping
        if f1_val > f1_best:
            f1_best = f1_val
            best_epoch = epoch
            save_path = os.path.join(config.model_path, f'{os.path.basename(config.Dataset_dir)}_best_model_edge_mask.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {epoch} to {save_path}")
            bad_counter = 0
        else:
            bad_counter += 1
            print(f"Epoch {epoch}: Early Stop Counter = {bad_counter}")
            if bad_counter >= config.early_stop_epochs:
                print(f"Early stopping triggered! Best Validation Accuracy: {f1_best:.4f} at epoch {best_epoch}")
                break

    total_time = time.time() - t_total
    print(f"Optimization Finished! Total time: {total_time:.2f}s")
    print(f"Best Validation Accuracy: {f1_best:.4f} at epoch {best_epoch}")
    return f1_best

def test(model, loader, config, device):
    """
    Gather model prediction probabilities and true labels for a DataLoader.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_aa = model.mlp_aa(batch.x_esm)
            x_pc = model.mlp_pc(batch.x_physio)
            x_labeled = x_aa + x_pc
            output = model(x_labeled, batch.edge_index)
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_probs, all_labels

# =========================
# Dataset Preparation
# =========================

dataset_dir = os.path.abspath('./ABP_data/')
seq_dir = os.path.join(dataset_dir, 'labeled_seq')
unlabeled_seq_dir = os.path.join(dataset_dir, 'unlabeled_seq') 
unlabeled_seq_files = [f for f in os.listdir(unlabeled_seq_dir) if f.endswith('.fasta')]
unlabeled_seq_ids = [os.path.splitext(f)[0] for f in unlabeled_seq_files]

dataset_files = {'train': '64_Train.txt', 'test': '14_Test.txt'}

# Load annotations and split dataset
train_list, test_list, seqanno = load_annotations(dataset_dir, dataset_files)
train_list, valid_list = tv_split(train_list, seed=1995)
StatisticsSampleNum(train_list, valid_list, test_list, seqanno)

seq_list = train_list + valid_list + test_list

# Load labeled pickle files
with open(os.path.join(dataset_dir, 'labeled_EI.pkl'), 'rb') as f:
    labeled_edge_indices = pickle.load(f)
with open(os.path.join(dataset_dir, 'labeled_pf.pkl'), 'rb') as f:
    labeled_physio_features_dict = pickle.load(f)
with open(os.path.join(dataset_dir, 'labeled_ef.pkl'), 'rb') as f:
    labeled_esm_features_dict = pickle.load(f)

# Load unlabeled pickle files
with open(os.path.join(dataset_dir, 'unlabeled_EI.pkl'), 'rb') as f:
    unlabeled_edge_indices = pickle.load(f)
with open(os.path.join(dataset_dir, 'unlabeled_pf.pkl'), 'rb') as f:
    unlabeled_physio_features_dict = pickle.load(f)
with open(os.path.join(dataset_dir, 'unlabeled_ef.pkl'), 'rb') as f:
    unlabeled_esm_features_dict = pickle.load(f)

# Define dataset splits
dataset_splits = {
    'train': train_list,
    'valid': valid_list,
    'test': test_list,
    'unlabeled': unlabeled_seq_ids
}

# Label status for each sequence
is_labeled = {seq_id: False for seq_id in unlabeled_seq_ids}
is_labeled.update({seq_id: True for seq_id in train_list + valid_list + test_list})

# Create PyG Data for labeled data
labeled_data = create_pyg_data_train(
    esm_features=labeled_esm_features_dict,
    physio_features=labeled_physio_features_dict,
    edge_indices=labeled_edge_indices,
    seqanno=seqanno,
    dataset_splits={'train': train_list, 'valid': valid_list, 'test': test_list},
    is_labeled=is_labeled
)

# Create PyG Data for unlabeled data
unlabeled_data = create_pyg_data_train(
    esm_features=unlabeled_esm_features_dict,
    physio_features=unlabeled_physio_features_dict,
    edge_indices=unlabeled_edge_indices,
    seqanno={},  # No annotations for unlabeled data
    dataset_splits={'unlabeled': unlabeled_seq_ids},
    is_labeled={seq_id: False for seq_id in unlabeled_seq_ids}
)['unlabeled']

# =========================
# Model, Optimizer, Training Setup
# =========================

config = Config()
config.print_config()

# DataLoaders
batch_size = config.batch_size
unlabeled_loader = DataLoader(unlabeled_data, batch_size=config.batch_size, shuffle=True)
train_loader = DataLoader(labeled_data['train'], batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(labeled_data['valid'], batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(labeled_data['test'], batch_size=config.batch_size, shuffle=False)

# Initialize model, optimizer, loss, scheduler
model = GAT(
    config=config,
    esm_input_dim=config.input_dim_esm,
    physio_input_dim=config.input_dim_physio,
    hidden_channels=config.hidden_channels,
    out_channels=config.out_channels,
    heads=config.heads,
    heads_intermediate=config.heads_intermediate,
    heads_final=config.heads_final,
    dropratio=config.dropratio
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config.str_lr, weight_decay=config.L2_weight)
criterion = FocalLoss(alpha=0.3, gamma=2.0, reduction='mean', num_classes=config.out_channels).to(device)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.9, patience=20, min_lr=1e-6
)

# Ensure model directory exists
os.makedirs(config.model_path, exist_ok=True)

# =========================
# Training
# =========================
print('===== Training Started =====')
best_val_acc = train(
    model=model, 
    optimizer=optimizer, 
    criterion=criterion, 
    consis=consis, 
    config=config, 
    train_loader=train_loader, 
    unlabeled_loader=unlabeled_loader, 
    val_loader=val_loader, 
    test_loader=test_loader, 
    device=device, 
    scheduler=scheduler
)

# =========================
# Final Evaluation
# =========================
best_model_path = os.path.join(config.model_path, f'{os.path.basename(config.Dataset_dir)}_best_model_edge_mask.pth')
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    print(f'Loaded best model from {best_model_path} with Validation Accuracy: {best_val_acc:.4f}')
else:
    print(f"No saved model found at {best_model_path}.")

print('===== Final Evaluation =====')

# Validation set
valid_probs, valid_labels = test(model, val_loader, config, device)
# Test set
test_probs, test_labels = test(model, test_loader, config, device)

# Compute metrics
acc_val, pre_val, rec_val, f1_val = eval_metrics(valid_probs, valid_labels, config.out_channels)
valid_matrices = acc_val, pre_val, rec_val, f1_val
acc_test, pre_test, rec_test, f1_test = eval_metrics(test_probs, test_labels, config.out_channels)
test_matrices = acc_test, pre_test, rec_test, f1_test

# Print results
print_results(valid_matrices, test_matrices)