#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from peft import PeftModel
from transformers import EsmTokenizer, EsmModel
from typing import Dict, Union
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta_to_dataframe(fasta_file):
    """
    Reads a multi-sequence FASTA file and converts it to a DataFrame.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        pd.DataFrame: DataFrame containing sequence IDs and sequences.
    """
    sequences = {'id': [], 'sequence': []}
    current_id = None
    current_sequence = ''

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences['id'].append(current_id)
                    sequences['sequence'].append(current_sequence)
                current_id = line[1:]
                current_sequence = ''
            else:
                current_sequence += line
        if current_id is not None:
            sequences['id'].append(current_id)
            sequences['sequence'].append(current_sequence)

    return pd.DataFrame(sequences)


def extract_features(
    model,
    inp: torch.Tensor,
    has_cls: bool = True,
    has_eos: bool = True,
    need_head_weights: Union[bool, Dict] = False,
):
    """
    Extract attention features from the model.

    Args:
        model: The transformer model.
        inp: Input tensor.
        has_cls: Whether the input has a CLS token.
        has_eos: Whether the input has an EOS token.
        need_head_weights: Whether head weights are needed.

    Returns:
        Attention features.
    """
    out_start_idx = 1 if has_cls else 0
    out_end_idx = -1 if has_eos else None
    result = model(inp, output_attentions=True, output_hidden_states=True)

    attentions = torch.stack(result.attentions, 1)
    batch, layer, head, seqlen, _ = attentions.size()
    attentions = attentions.reshape(batch, layer * head, seqlen, seqlen)

    output_attentions = attentions.detach()
    output_attentions.requires_grad = False

    return output_attentions[:, :, out_start_idx:out_end_idx, out_start_idx:out_end_idx]


def _process_output(out_dict):
    """
    Process the output dictionary from the model.

    Args:
        out_dict: Dictionary containing model outputs.

    Returns:
        Processed output dictionary.
    """
    out_dict['logits'] = out_dict['logits'].permute(0, 2, 3, 1).contiguous()

    if 'logits' in out_dict:
        out_dict['dist_logits'] = out_dict['logits']
        del out_dict['logits']
    for k in [k for k in out_dict if k.endswith('_logits')]:
        targetname = k[:-len('_logits')]
        out_dict[f'p_{targetname}'] = torch.sigmoid(out_dict[k])
    return out_dict


class ResidualBlock(nn.Module):
    """
    Residual block with convolutional layers and batch normalization.
    """

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
    """
    Model to predict contact maps using residual projection architecture.
    """

    def __init__(self, base_model, num_heads=20 * 33, output_dim=1, channels=[512, 256, 128, 64]):
        super().__init__()
        layers = []
        in_channels = num_heads

        for out_channels in channels:
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_channels, output_dim, kernel_size=1)

    def forward(self, inp):
        attentions_asym = self.extract_features(inp)

        out = self.feature_extractor(attentions_asym)
        out = self.final_conv(out)
        return {'logits': self.symmetrize(out)}

    def extract_features(self, inp):
        return extract_features(base_model, inp, need_head_weights=True)

    @staticmethod
    def symmetrize(x, scale=1.0):
        return scale * (x + x.transpose(-1, -2))


def adj_to_edge_index(adj):
    """
    Convert adjacency matrix to edge index format.

    Args:
        adj: Adjacency matrix.

    Returns:
        Edge index tensor.
    """
    src, dst = np.nonzero(adj)
    return torch.stack([torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)], dim=0)


def Create_Contacts_EdgeIndex(seq_df, ei_dir, base_model, contact_model):
    """
    Create contact maps and save them as edge indices.

    Args:
        seq_df: DataFrame containing sequence IDs and sequences.
        ei_dir: Directory to save edge index files.
        base_model: Base model for feature extraction.
        contact_model: Path to contact prediction model weights.
    """
    if not os.path.exists(ei_dir):
        os.makedirs(ei_dir)

    model = ResidualProjectionDistogramModel(
        base_model,
        num_heads=20 * 33,
        output_dim=1,
        channels=[512, 256, 128, 64, 32, 16]
    )

    model.load_state_dict(torch.load(contact_model))
    model.to(device)
    contact_threshold = 0.65  # Threshold for contact prediction

    for i, s in zip(seq_df.id, seq_df.sequence):
        inputs = tokenizer(s, return_tensors='pt', padding="max_length", truncation=True, max_length=len(s) + 2).to(device)

        with torch.no_grad():
            out_dict = model(inputs['input_ids'])
            outputs = _process_output(out_dict)

        contacts = (outputs['p_dist'] > contact_threshold).float()[0, :, :, 0]
        contacts = np.nan_to_num(contacts.cpu().detach().numpy(), nan=0)
        contacts = np.clip(contacts, 0, 1)
        edge_index = adj_to_edge_index(contacts)

        edge_index_path = os.path.join(ei_dir, f"{i}_ei.pkl")
        with open(edge_index_path, 'wb') as f:
            pickle.dump(edge_index.numpy(), f)

#     print(f"All edge indices have been successfully saved in {ei_dir}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process FASTA files and create edge indices.")
    parser.add_argument('--fasta_file', type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save all output files in subdirectories: ei_dir.")
    parser.add_argument("--cm", type=str, required=True, help="Path to contact prediction model weights.")
    parser.add_argument("--plm", type=str, required=True, help="Path or model name to protein language model weights (ESM2).")

    args = parser.parse_args()
    fasta_file = args.fasta_file
    output_dir = args.output_dir
    contact_model = args.cm
    model_checkpoint = args.plm

    if not os.path.isfile(fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    ei_dir = os.path.join(output_dir, "ei_dir")
    os.makedirs(ei_dir, exist_ok=True)

    seq_df = read_fasta_to_dataframe(fasta_file)
    tokenizer = EsmTokenizer.from_pretrained(model_checkpoint)
    base_model = EsmModel.from_pretrained(model_checkpoint).to(device)

    Create_Contacts_EdgeIndex(seq_df, ei_dir, base_model, contact_model)
#     print("Edge indices calculation completed.")