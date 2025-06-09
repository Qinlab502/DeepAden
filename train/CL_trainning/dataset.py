# dataset.py
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ContrastiveDataset(Dataset):
    def __init__(self, csv_file, all_molecules_csv=None, protein_feature_dir="data/protein_data", 
                 molecule_feature_dir="data/molecule_data"):
        """
        Dataset for contrastive learning using protein and molecule data.
        
        Args:
            csv_file: Path to CSV file containing 'id', 'pocket' and 'labels' columns
            all_molecules_csv: Path to CSV file containing all molecule information (if None, use unique molecules from csv_file)
            protein_feature_dir: Directory containing precomputed protein features
            molecule_feature_dir: Directory containing precomputed molecule features
        """
        self.df = pd.read_csv(csv_file)
        self.protein_feature_dir = protein_feature_dir
        self.molecule_feature_dir = molecule_feature_dir
        
        # Check if required columns exist
        required_cols = ['id', 'labels']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV file missing required columns: {missing_cols}")
        
        # Load the complete list of all molecules if provided
        if all_molecules_csv:
            self.all_molecules_df = pd.read_csv(all_molecules_csv)
            if 'labels' not in self.all_molecules_df.columns:
                raise ValueError("all_molecules_csv must contain a 'labels' column")
            self.all_molecules = self.all_molecules_df['labels'].unique()
            print(f"Loaded {len(self.all_molecules)} total molecules from {all_molecules_csv}")
        else:
            # Use only molecules from the input CSV if no complete list is provided
            self.all_molecules = self.df['labels'].unique()
            print(f"Using {len(self.all_molecules)} molecules from input CSV")
        
        # Create comprehensive set of pairs
        self._create_all_pairs()
    
    def _create_all_pairs(self):
        """Create pairs between each protein in the input data and ALL molecules"""
        # Get unique proteins from the input data
        all_proteins = self.df['id'].unique()
        
        print(f"Found {len(all_proteins)} unique proteins and will pair with {len(self.all_molecules)} molecules")
        
        # Create a dictionary to track positive pairs (binding interactions)
        positive_pairs = {}
        for _, row in self.df.iterrows():
            protein_id = row['id']
            molecule_label = row['labels']
            positive_pairs[(protein_id, molecule_label)] = True
        
        print(f"Found {len(positive_pairs)} positive binding pairs")
        
        # Create all possible pairs between input proteins and ALL molecules
        self.pairs = []
        total_possible_pairs = len(all_proteins) * len(self.all_molecules)
        
        for protein_id in all_proteins:
            for molecule_label in self.all_molecules:
                # Check if this is a positive pair (binding interaction)
                is_positive = (protein_id, molecule_label) in positive_pairs
                
                self.pairs.append({
                    'protein_id': protein_id,
                    'molecule_label': molecule_label,
                    'label': 1.0 if is_positive else 0.0
                })
        
        positive_count = sum(1 for p in self.pairs if p['label'] > 0.5)
        negative_count = sum(1 for p in self.pairs if p['label'] < 0.5)
        
        print(f"Created {len(self.pairs)} total pairs:")
        print(f"  - {positive_count} positive pairs ({positive_count/len(self.pairs)*100:.2f}%)")
        print(f"  - {negative_count} negative pairs ({negative_count/len(self.pairs)*100:.2f}%)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        protein_id = pair['protein_id']
        molecule_label = pair['molecule_label']
        interaction_label = pair['label']
        
        # Load precomputed features
        protein_feature_path = os.path.join(self.protein_feature_dir, f"{protein_id}.pt")
        if not os.path.exists(protein_feature_path):
            raise FileNotFoundError(f"Protein feature file not found: {protein_feature_path}")
        protein_feature = torch.load(protein_feature_path, weights_only=True)
        
        molecule_feature_path = os.path.join(self.molecule_feature_dir, f"{molecule_label}.pt")
        if not os.path.exists(molecule_feature_path):
            raise FileNotFoundError(f"Molecule feature file not found: {molecule_feature_path}")
        molecule_feature = torch.load(molecule_feature_path, weights_only=True)
        
        return {
            'protein_id': protein_id,
            'molecule_label': molecule_label,
            'protein_feature': protein_feature,
            'molecule_feature': molecule_feature,
            'label': torch.tensor(interaction_label, dtype=torch.float)
        }

def split_dataset(csv_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split the dataset into train, validation and test sets.
    
    Args:
        csv_file: Path to the original CSV file
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Paths to the created train, validation and test CSV files
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get unique protein IDs to ensure same protein doesn't appear in different splits
    unique_proteins = df['id'].unique()
    
    # Shuffle protein IDs
    np.random.seed(random_seed)
    np.random.shuffle(unique_proteins)
    
    # Calculate split sizes
    train_size = int(len(unique_proteins) * train_ratio)
    val_size = int(len(unique_proteins) * val_ratio)
    
    # Split protein IDs
    train_proteins = unique_proteins[:train_size]
    val_proteins = unique_proteins[train_size:train_size + val_size]
    test_proteins = unique_proteins[train_size + val_size:]
    
    # Create dataframes for each split
    train_df = df[df['id'].isin(train_proteins)]
    val_df = df[df['id'].isin(val_proteins)]
    test_df = df[df['id'].isin(test_proteins)]
    
    # Save to CSV files
    base_dir = os.path.dirname(csv_file)
    train_file = os.path.join(base_dir, "train_split.csv")
    val_file = os.path.join(base_dir, "val_split.csv")
    test_file = os.path.join(base_dir, "test_split.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Split dataset: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")
    print(f"Split by proteins: {len(train_proteins)} train, {len(val_proteins)} validation, {len(test_proteins)} test proteins")
    
    return train_file, val_file, test_file