import os
import esm
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.pdb_loader as pdb_loader

from tqdm import tqdm
from torch.optim import AdamW
from collections import namedtuple
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict,Optional, List, Callable, Union
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, matthews_corrcoef
from transformers import EsmTokenizer, EsmModel, EsmConfig,EsmForMaskedLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_layers=33
attention_heads=20
model_name = "../esm_masked_training/facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
base_model = EsmForMaskedLM.from_pretrained(model_name).to(device)
base_model.eval()
base_model.to(device)

def extract_features(
    model,
    inp: torch.Tensor,
    has_cls: bool = True,
    has_eos: bool = True,
    need_head_weights: Union[bool, Dict] = False,
):
    """
    Inflexible way of dealing with the mess of inputs with cls tokens,
    need embedding without, and LSTM doesnt want cls token in the first place.
    """
    
    out_start_idx = 1 if has_cls else 0
    out_end_idx = -1 if has_eos else None
    result = model(**inp, output_attentions=True, output_hidden_states=True)
    
    attentions = result["attentions"]
    attentions=torch.stack(attentions,1)

    batch, layer, head, seqlen, seqlen2 = attentions.size()
    assert seqlen == seqlen2
    attentions = attentions.reshape(
        batch, layer * head, seqlen, seqlen
    )
    emb = result["logits"]
    
    output_attentions=attentions.detach()
    output_attentions.requires_grad=False
    
    
    
    return (emb[:, out_start_idx:out_end_idx], output_attentions[:, :, out_start_idx:out_end_idx, out_start_idx:out_end_idx])

def _process_output(out_dict):
    out_dict['logits'] = out_dict['logits'].permute(0, 2, 3, 1).contiguous()
    
    if 'logits' in out_dict:
        out_dict['dist_logits'] = out_dict['logits']
        del out_dict['logits']
    for k in [k for k in out_dict if k.endswith('_logits')]:
        targetname = k[:-len('_logits')]
        out_dict[f'p_{targetname}'] = torch.sigmoid(out_dict[k])
    return out_dict

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ResidualProjectionDistogramModel(nn.Module):
    def __init__(self, base_model, num_heads=20*33, output_dim=1, channels=[512, 256, 128, 64]):
        super().__init__()
        layers = []
        in_channels = num_heads
        
        for out_channels in channels:
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_channels, output_dim, kernel_size=1)

    def forward(self, inp):
        _, attentions_asym = self.extract_features(inp)
        
        out = self.feature_extractor(attentions_asym)
        out = self.final_conv(out)
        out = self.symmetrize(out)
        
        return {'logits': out}

    def extract_features(self, inp):
        return extract_features(base_model, inp, need_head_weights=True)

    @staticmethod
    def symmetrize(x, scale=1.0):
        return scale * (x + x.transpose(-1, -2))


RPDmodel = ResidualProjectionDistogramModel(
    base_model=base_model,
    num_heads=20*33, 
    output_dim=1, 
    channels=[512, 256, 128, 64, 32, 16]
)

PDB_LOADER_PARAM_REGISTRY = {
    'LinearProjectionDist-1A': {
        "DMIN": 2.5,
        "DMAX": 20.0,
        "DBINS": 17,
        "ABINS": 17,
        "PHI_BINS": 7,
        "WMIN": 0.8,
        "LMIN": 150,
        "LMAX": 650,  
        "EPOCHS": 10,
        "NCPU": 8,
        "SLICE": "CONT",
        "contact_bin_cutoff": (0, 5)
    },
}

        
pdb_loader_params = PDB_LOADER_PARAM_REGISTRY['LinearProjectionDist-1A']

target_datas = {}
pdb_directory = './training_full_seqs_data/pdb/'
pdb_files_path = [os.path.join(pdb_directory, i) for i in os.listdir(pdb_directory)]

for pdb_file in pdb_files_path:
    if pdb_file.endswith('pdb'):
        try:
            target_data = pdb_loader.loader(
                          pdb_path=pdb_file,
                          params=pdb_loader_params,
                          set_diagonal=True,
                          allow_missing_residue_coords=True)
            target_data = np.where(target_data['dist'] < 8, 1, 0)
            target_datas[os.path.basename(pdb_file)[:-4]] = target_data
        except Exception as e:
            print(f"Error reading pdb_file: {pdb_file}")
            continue
            
def get_fasta_dict(fasta_file):
    fasta_dict = {}
    with open(fasta_file, 'r') as infile:
        for line in infile:
            if line.startswith(">"):
                head = line.strip().replace(">", "")
                fasta_dict[head] = ''
            else:
                fasta_dict[head] += line.replace("\n", "")
    return fasta_dict

def get_label(fasta_file):
    labels=[]
    for header, sequence in fasta_dict.items():
        header=header
        label=torch.tensor(target_datas[header])
        labels.append(label)
    return labels


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_dict) -> None:
        super().__init__()
        self.fasta_dict = fasta_dict
        self.names = [name for name in fasta_dict.keys()]
        self.labels = get_label(fasta_dict)
        
    def __getitem__(self, idx):
        seq_name = self.names[idx]
        selected_seq = self.fasta_dict[seq_name]
        label = self.labels[idx]
        
        return seq_name, selected_seq, label
    
    def __len__(self):
        return len(self.names)
    
fasta_dict = get_fasta_dict('./training_full_seqs_data/80_seqs.fasta')
dataset = SeqDataset(fasta_dict)
print("Dataset checking:")
for i in tqdm(range(dataset.__len__())):
    seq_name, seq, label = dataset.__getitem__(i)
    
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

num_epochs = 1
lr = 1e-3

base_model.to(device)
model = RPDmodel.to(device)

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.8, min_lr=1e-6)
loss_fn = nn.BCELoss()  

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def eval_metrics(probs, targets, num_classes):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if probs.ndim == 1:
        pred = probs
    else:
        pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(targets, pred, labels=np.arange(num_classes))
    precision, recall, f1_score, _ = precision_recall_fscore_support(targets, pred, average='macro', zero_division=1)
    accuracy = np.trace(cm) / np.sum(cm)
    mcc = matthews_corrcoef(targets, pred)
    
    return accuracy, precision, recall, f1_score, mcc

def find_best_threshold(model, dataset, thresholds):
    best_threshold = 0
    best_mcc = 0

    for threshold in thresholds:
        acc_list = []
        pre_list = []
        rec_list = []
        F1_list = []
        mcc_list = []

        model.eval()

        for i, (n, s, l) in enumerate(dataset):
            inputs = tokenizer(s, return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = l.to(device)

            with torch.no_grad():
                out_dict = model(inputs)
                outputs = _process_output(out_dict)

                target = labels.ravel()
                predicted = (outputs['p_dist'] > threshold).float().ravel()

                val_acc, val_pre, val_rec, val_F1, val_mcc = eval_metrics(predicted, target, 2)
                acc_list.append(val_acc)
                pre_list.append(val_pre)
                rec_list.append(val_rec)
                F1_list.append(val_F1)
                mcc_list.append(val_mcc)

        total_mcc = np.mean(np.array(mcc_list)) 
        if total_mcc > best_mcc:
            best_mcc = total_mcc
            best_threshold = threshold

    return best_threshold

def val(model, dataset, threshold=None):
    if threshold is None:
        best_threshold = find_best_threshold(model, dataset, np.arange(0, 1.02, 0.05))
    else:
        best_threshold = threshold

    acc_list = []
    pre_list = []
    rec_list = []
    F1_list = []
    mcc_list = []

    model.eval()

    for i, (n, s, l) in enumerate(dataset):
        inputs = tokenizer(s, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = l.to(device)

        with torch.no_grad():
            out_dict = model(inputs)
            outputs = _process_output(out_dict)

            target = labels.ravel()
            predicted = (outputs['p_dist'] > best_threshold).float().ravel()

            val_acc, val_pre, val_rec, val_F1, val_mcc = eval_metrics(predicted, target, 2)
            acc_list.append(val_acc)
            pre_list.append(val_pre)
            rec_list.append(val_rec)
            F1_list.append(val_F1)
            mcc_list.append(val_mcc)

    total_acc = np.mean(np.array(acc_list))
    total_pre = np.mean(np.array(pre_list))
    total_rec = np.mean(np.array(rec_list))
    total_F1 = np.mean(np.array(F1_list))
    total_mcc = np.mean(np.array(mcc_list))

    return total_acc, total_pre, total_rec, total_F1, total_mcc, best_threshold


train_loss_list = []
train_acc_list = []
train_F1_list = []
test_acc_list = []
test_F1_list = []


best_mcc=0 

progress_bar = tqdm(total=num_epochs)

for epoch in range(num_epochs):
    # Train the model on the training set
    train_loss = 0.0
    model.train()

    for i, (n, s, l) in enumerate(train_dataset):
        
        inputs = tokenizer(s, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = l.to(device)
        optimizer.zero_grad()# Reset gradients
        out_dict = model(inputs)
        outputs = _process_output(out_dict)

        loss = loss_fn(outputs['p_dist'][0,:,:,0], labels.float())
        train_loss += loss


        loss.backward()# calulate loss
        optimizer.step()# update gradient
        
        
    train_loss = train_loss / len(train_dataset)
    train_loss = train_loss.detach().cpu().tolist()
    
    train_loss_list.append(train_loss)
    
    
    test_acc, test_pre, test_rec, test_F1, test_mcc, threshold = val(model, test_dataset, threshold=0.5)  #glj 730
    test_acc_list.append(test_acc)
    test_F1_list.append(test_F1)
    scheduler.step(test_mcc) 
    
    progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
    progress_bar.set_postfix(loss=f'{train_loss:.4f}', lr=f"{optimizer.param_groups[0]['lr']:.4f}", test_mcc=f'{test_mcc:.4f}', test_F1=f'{test_F1:.4f}', test_acc=f'{test_acc:.4f}', threshold=f'{threshold:.2f}')
    
    
    if test_mcc > best_mcc:
        best_state = copy.deepcopy(model.state_dict())
        best_mcc = test_mcc
        best_threshold = threshold
        best_F1 = test_F1
    
    progress_bar.update(1)

torch.save(best_state, './best_model.pt')