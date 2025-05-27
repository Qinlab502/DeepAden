#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# ============================
# Utility Functions
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta_to_dataframe(fasta_file: str) -> pd.DataFrame:
    """
    Read a multi-sequence FASTA file and convert it to a DataFrame.

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

    df = pd.DataFrame(sequences)
    return df


def def_residue_feature() -> pd.DataFrame:
    """
    Define and return a DataFrame containing physicochemical features of amino acid residues.

    Returns:
        pd.DataFrame: DataFrame with each row representing an amino acid and columns as features.
    """
    residue_feature = {
        'G': [0.0, 75, 5.97, 0, 60, 0, 0, 0.48, 1, 75, 1, 1, 0, 0, 0, 0.61, 1.2, 2.34],
        'A': [0.31, 89, 6.01, 0, 88, 0, 0, 0.62, 1, 115, 1, 0, 0, 0, 0, 0.35, 1.5, 2.35],
        'V': [1.22, 117, 5.96, 0, 140, 0, 0, 0.86, 1, 162, 0, 0, 0, 0, 0, 0.43, 2.0, 2.32],
        'L': [1.70, 131, 5.98, 0, 170, 0, 0, 0.97, 1, 167, 0, 0, 0, 0, 0, 0.29, 2.2, 2.36],
        'I': [1.80, 131, 6.02, 0, 170, 0, 0, 0.97, 1, 168, 0, 0, 0, 0, 0, 0.43, 2.2, 2.36],
        'P': [0.72, 115, 6.3, 0, 129, 0, 0, 0.64, 0, 145, 0, 0, 0, 0, 1, 0.55, 1.8, 1.99],
        'F': [1.79, 165, 5.48, 0, 210, 0, 1, 1.13, 0, 190, 0, 0, 1, 0, 0, 0.22, 2.3, 2.20],
        'Y': [0.96, 181, 5.66, 0, 230, 0, 1, 1.09, 0, 200, 0, 0, 1, 0, 0, 0.28, 2.5, 2.20],
        'W': [2.25, 204, 5.89, 0, 250, 0, 1, 1.35, 0, 237, 0, 0, 1, 0, 0, 0.20, 2.8, 2.38],
        'S': [-0.04, 105, 5.68, 0, 100, 1, 0, 0.51, 1, 140, 1, 0, 0, 0, 0, 0.42, 1.8, 2.19],
        'T': [0.26, 119, 5.6, 0, 120, 1, 0, 0.61, 1, 150, 1, 0, 0, 0, 0, 0.36, 1.9, 2.25],
        'C': [1.54, 121, 5.07, 0, 150, 1, 0, 0.97, 0, 170, 0, 0, 0, 1, 0, 0.55, 1.7, 2.05],
        'M': [1.23, 149, 5.74, 0, 160, 0, 0, 0.85, 0, 185, 0, 0, 0, 1, 0, 0.46, 2.0, 2.28],
        'N': [-0.60, 132, 5.41, 0, 135, 1, 0, 0.42, 1, 155, 1, 0, 0, 0, 0, 0.53, 1.9, 2.18],
        'Q': [-0.22, 146, 5.65, 0, 145, 1, 0, 0.58, 1, 165, 1, 0, 0, 0, 0, 0.44, 2.1, 2.17],
        'D': [-0.77, 133, 2.77, -1, 130, 1, 0, 0.37, 1, 150, 1, -1, 0, 0, 0, 0.60, 1.6, 1.88],
        'E': [-0.64, 147, 3.22, -1, 150, 1, 0, 0.43, 1, 170, 1, -1, 0, 0, 0, 0.46, 1.8, 1.91],
        'K': [-0.99, 146, 9.74, 1, 170, 1, 0, 0.33, 1, 200, 0, 1, 0, 0, 0, 0.32, 2.6, 2.18],
        'R': [-1.01, 174, 10.76, 1, 210, 1, 0, 0.21, 1, 225, 0, 1, 0, 0, 0, 0.20, 2.8, 2.17],
        'H': [0.13, 155, 7.59, 0, 180, 1, 1, 0.6, 0, 195, 1, 0, 1, 0, 0, 0.44, 2.3, 1.82],
        'X': [0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    columns = [
        'hydrophobicity', 'molecular_weight', 'isoelectric_point',
        'charge', 'volume', 'polarity', 'aromaticity',
        'kyte_doolittle_hydropathy', 'flexibility', 'surface_area',
        'alpha_helix_tendency', 'beta_sheet_tendency', 'charge_class',
        'family_class', 'reactivity', 'solvent_accessibility',
        'covalent_radius', 'pKa'
    ]

    residue_df = pd.DataFrame.from_dict(residue_feature, orient='index', columns=columns)
    residue_df.index.name = 'residue'
    residue_df.reset_index(inplace=True)

    # Min-Max Scaling
    for column in residue_df.columns[1:]:
        min_val = residue_df[column].min()
        max_val = residue_df[column].max()
        residue_df[column] = (residue_df[column] - min_val) / (max_val - min_val)

    return residue_df


def get_seq_features(sequence, residue_features):
    """
    Extract features for each residue in a sequence.

    Args:
        sequence (str): Amino acid sequence.
        residue_features (pd.DataFrame): DataFrame containing residue features.

    Returns:
        Tuple[np.ndarray, List[int]]: Array of physicochemical features and list of residue indices.
    """
    sequence_features = []
    res_id_list = []

    for idx, residue in enumerate(sequence):
        if residue in residue_features.index:
            residue_row = residue_features.loc[residue].values.tolist()
            sequence_features.append(residue_row)
        else:
            print(f"Unknown residue '{residue}' encountered.")
            sequence_features.append([0.0] * len(residue_features.columns))

        res_id_list.append(idx + 1)

    # Convert to numpy array for efficiency
    sequence_features = np.array(sequence_features, dtype=np.float32)

    return sequence_features, res_id_list


# ============================
# Feature Calculation Functions
# ============================

def calculate_features(sequences_dict: Dict[str, str], pf_dir: str, emb_dir: str,
                      esm_model: EsmModel, esm_tokenizer: EsmTokenizer, device: torch.device) -> None:
    """
    Calculate physicochemical features and ESM2 embeddings for all sequences and save them as pickle files.

    Args:
        sequences_dict (Dict[str, str]): Dictionary with sequence IDs as keys and sequence strings as values.
        pf_dir (str): Directory to save physicochemical feature pickle files.
        emb_dir (str): Directory to save ESM2 embedding pickle files.
        esm_model (EsmModel): Pretrained ESM2 model.
        esm_tokenizer (EsmTokenizer): Pretrained ESM2 tokenizer.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    os.makedirs(pf_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    residue_features = def_residue_feature()
    residue_features.set_index('residue', inplace=True)

    esm_model.eval()

#     print("Processing sequences...")
    for seq_id, sequence in sequences_dict.items():
        # Calculate physicochemical features
        physio_features, res_id_list = get_seq_features(sequence, residue_features)
        physio_pickle_path = os.path.join(pf_dir, f"{seq_id}_pf.pkl")
        with open(physio_pickle_path, 'wb') as f:
            pickle.dump(physio_features, f)

        # Calculate ESM2 embeddings
        inputs = esm_tokenizer(sequence, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = esm_model(**inputs, output_attentions=False, output_hidden_states=False)
            representations = outputs.last_hidden_state.squeeze(0)  # [seq_len, embedding_dim]

        # Remove CLS and EOS tokens
        if representations.shape[0] < 2:
            print(f"Sequence {seq_id} is too short for ESM2 processing.")
            esm_embeddings = np.zeros((0, representations.shape[1]), dtype=np.float32)
        else:
            esm_embeddings = representations[1:-1].cpu().numpy()  # [residues, embedding_dim]

        esm_pickle_path = os.path.join(emb_dir, f"{seq_id}_emb.pkl")
        with open(esm_pickle_path, 'wb') as f:
            pickle.dump(esm_embeddings, f)

#     print(f"Processed {len(sequences_dict)} sequences successfully.")


# ============================
# Main Function
# ============================

def main():
    parser = argparse.ArgumentParser(description="Process sequences from a FASTA file and compute physiochemical features and ESM2 embeddings.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input multi-sequence FASTA file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save all output files in subdirectories: pf_dir, emb_dir.")
    parser.add_argument("--plm", type=str, required=True, help="Path or model name to protein language model weights (ESM2).")

    args = parser.parse_args()

    fasta_file = args.fasta_file
    output_dir = args.output_dir
    plm_path = args.plm

    # Check if input file exists
    if not os.path.isfile(fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    # Define subdirectory paths
    pf_dir = os.path.join(output_dir, "pf_dir")
    emb_dir = os.path.join(output_dir, "emb_dir")

    # Create all subdirectories if they don't exist
    sub_dirs = [pf_dir, emb_dir]
    for directory in sub_dirs:
        os.makedirs(directory, exist_ok=True)
#         print(f"Output directory: {directory}")

    # Step 1: Read FASTA file
#     print("Reading FASTA file...")
    seq_df = read_fasta_to_dataframe(fasta_file)
    sequences_dict = dict(zip(seq_df['id'], seq_df['sequence']))
#     print(f"Found {len(sequences_dict)} sequences in the input file.")

    # Step 2: Initialize ESM2 model
#     print("Initializing ESM2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_esm = EsmTokenizer.from_pretrained(plm_path)
    model_esm = EsmModel.from_pretrained(plm_path).to(device)
    model_esm.eval()

    # Step 3: Calculate features
    calculate_features(sequences_dict, pf_dir, emb_dir, model_esm, tokenizer_esm, device)

#     print("Processing completed successfully.")


if __name__ == "__main__":
    main()