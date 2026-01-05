# ================================================================
# dataset.py — Simple supervised contrastive dataset (name-level + index label)
# ================================================================
import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class ContrastiveDataset(Dataset):
    """
    Modified version:
    - Each row links a protein to its molecule (by molecule name).
    - Returns both molecule name (string) and its index (int).
    """

    def __init__(self, csv_file, protein_feature_dir, molecule_feature_dir):
        """
        Args:
            csv_file (str): Must contain 'id' (protein_id) and 'label' (molecule_name)
            protein_feature_dir (str): Path of protein .pt files
            molecule_feature_dir (str): Path of molecule .pt files
        """
        self.df = pd.read_csv(csv_file)
        self.protein_feature_dir = protein_feature_dir
        self.molecule_feature_dir = molecule_feature_dir

        # --- Column check ---
        required_cols = {"id", "label"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError("CSV must contain ['id', 'label'] columns")

        # --- Build mapping from molecule_name to index ---
        unique_molecules = sorted(self.df["label"].unique())
        self.molecule_to_index = {mol_name: idx for idx, mol_name in enumerate(unique_molecules)}

        print(f"[Dataset] Detected {len(unique_molecules)} unique molecules.")
        print(f"[Dataset] Total samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            {
                'protein_id': str,
                'molecule_id': str,
                'protein_feature': Tensor,
                'molecule_feature': Tensor,
                'label': str,              # molecule name (as string)
                'label_index': Tensor(long) # numeric index for contrastive loss
            }
        """
        row = self.df.iloc[idx]
        protein_id = str(row["id"])
        molecule_name = str(row["label"])

        # --- Load protein feature ---
        protein_path = os.path.join(self.protein_feature_dir, f"{protein_id}.pt")
        if not os.path.exists(protein_path):
            raise FileNotFoundError(f"Missing protein feature: {protein_path}")
        protein_feat = torch.load(protein_path, weights_only=True)

        # --- Load molecule feature ---
        molecule_path = os.path.join(self.molecule_feature_dir, f"{molecule_name}.pt")
        if not os.path.exists(molecule_path):
            raise FileNotFoundError(f"Missing molecule feature: {molecule_path}")
        molecule_feat = torch.load(molecule_path, weights_only=True)

        # --- Lookup numeric label index ---
        molecule_index = self.molecule_to_index[molecule_name]

        return {
            "protein_id": protein_id,
            "molecule_id": molecule_name,
            "protein_feature": protein_feat,
            "molecule_feature": molecule_feat,
            "label": molecule_name,  # string label for evaluation
            "label_index": torch.tensor(molecule_index, dtype=torch.long),  # numeric label for training
        }