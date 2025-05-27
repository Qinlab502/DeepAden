#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pickle
from torch_geometric.data import Data
import argparse

def create_pyg_data_objects(output_dir: str) -> None:
    """
    Create PyTorch Geometric Data objects from precomputed node features 
    (ESM embeddings and physicochemical features) and edge indices.

    Args:
        output_dir (str): Base directory containing emb_dir, pf_dir, ei_dir, 
                          and where feature_dir will be created.
    """
    # Define input and output directories
    esm_dir = os.path.join(output_dir, 'emb_dir')
    pf_dir = os.path.join(output_dir, 'pf_dir')
    ei_dir = os.path.join(output_dir, 'ei_dir')
    feature_dir = os.path.join(output_dir, 'feature_dir')

    # Ensure the output directory exists
    os.makedirs(feature_dir, exist_ok=True)

    # Get sequence IDs by listing embedding files
    seq_ids = set(
        file[:file.rfind('_')] for file in os.listdir(esm_dir) if file.endswith('_emb.pkl')
    )

    # Process each sequence ID
    for seq_id in seq_ids:
        esm_path = os.path.join(esm_dir, f"{seq_id}_emb.pkl")
        pf_path = os.path.join(pf_dir, f"{seq_id}_pf.pkl")
        ei_path = os.path.join(ei_dir, f"{seq_id}_ei.pkl")

        # Skip if any required file does not exist
        if not (os.path.exists(esm_path) and os.path.exists(pf_path) and os.path.exists(ei_path)):
            print(f"Warning: Sequence {seq_id} is missing features or edge index. Skipping.")
            continue

        # Load embeddings, physicochemical features, and edge indices
        with open(esm_path, 'rb') as f:
            esm_emb = pickle.load(f)

        with open(pf_path, 'rb') as f:
            physio_feat = pickle.load(f)

        with open(ei_path, 'rb') as f:
            edge_idx = pickle.load(f)

        # Create PyTorch Geometric Data object
        data = Data(
            x_esm=torch.tensor(esm_emb, dtype=torch.float),
            x_physio=torch.tensor(physio_feat, dtype=torch.float),
            edge_index=torch.tensor(edge_idx, dtype=torch.long)
        )
        data.num_nodes = data.x_esm.shape[0]

        # Save the Data object as a .pt file
        output_pt_path = os.path.join(feature_dir, f"{seq_id}.pt")
        torch.save(data, output_pt_path)

#     print(f"Completed processing all sequences. Output saved in '{feature_dir}'.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Create PyTorch Geometric Data objects from precomputed node features and edge indices."
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help="Base directory for embeddings, physicochemical features, edge indices, and output .pt files."
    )

    args = parser.parse_args()

    # Call the main function
    create_pyg_data_objects(args.input_dir)