import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import pickle
from model import ContrastiveModel

class BandwidthCallable:
    """
    A picklable class used to replace lambda functions.
    It simply returns the stored bandwidth value.
    """
    def __init__(self, value):
        self.value = value
    
    def __call__(self):
        return self.value

class KDECalibrator:
    """
    Calibrator based on Kernel Density Estimation (KDE).
    It models the distribution of cosine similarities for positive and negative pairs
    to calculate the posterior probability P(Positive | Similarity).
    """
    def __init__(self):
        self.kde_pos = None
        self.kde_neg = None
        self.prior_pos = 0.5  # Assuming balanced prior, can be adjusted based on application
        self.smooth_factor = 2
        
    def fit(self, pos_sims, neg_sims):
        """
        Fit KDE models for positive and negative distributions.
        """
        pos_data = np.array(pos_sims)
        neg_data = np.array(neg_sims)
        
        # Remove NaNs
        pos_data = pos_data[~np.isnan(pos_data)]
        neg_data = neg_data[~np.isnan(neg_data)]

        if len(pos_data) < 2 or len(neg_data) < 2:
            raise ValueError(f"Insufficient samples (Pos: {len(pos_data)}, Neg: {len(neg_data)}) to fit KDE.")

        print(f"[Calibrator] Fitting KDE with {len(pos_data)} positive and {len(neg_data)} negative samples.")

        # 'scott' rule is used for bandwidth selection
        self.kde_pos = gaussian_kde(pos_data, bw_method='scott')
        self.kde_neg = gaussian_kde(neg_data, bw_method='scott')
        
        bw_pos = self.kde_pos.factor * self.smooth_factor
        bw_neg = self.kde_neg.factor * self.smooth_factor
        
        self.kde_pos.covariance_factor = BandwidthCallable(bw_pos)
        self.kde_neg.covariance_factor = BandwidthCallable(bw_neg)
        
        self.kde_pos._compute_covariance()
        self.kde_neg._compute_covariance()

    def predict_proba(self, cosine_sim):
        """
        Convert cosine similarity to probability using Bayes' theorem.
        P(Pos|Sim) = (P(Sim|Pos) * P(Pos)) / P(Sim)
        """
        if self.kde_pos is None:
            raise RuntimeError("Calibrator not fitted yet!")

        if isinstance(cosine_sim, (float, int)):
            cosine_sim = np.array([cosine_sim])
        
        likelihood_pos = self.kde_pos(cosine_sim)
        likelihood_neg = self.kde_neg(cosine_sim)
        
        # Evidence: P(Sim)
        evidence = likelihood_pos * self.prior_pos + likelihood_neg * (1 - self.prior_pos)
        
        # Posterior: P(Pos|Sim)
        posterior = (likelihood_pos * self.prior_pos) / (evidence + 1e-10)
        
        return posterior

    def collect_data_full_retrieval(self, val_df, prot_dir, mol_dir, model, device, batch_size=128):
        """
        Optimized data collection using Full Retrieval strategy.
        
        1. Pre-loads all molecule features into memory (Matrix).
        2. For each protein, computes similarity against ALL molecules.
        3. Extracts the score for the true pair (Positive) and all others (Negatives).
        
        Args:
            val_df: DataFrame containing ['id', 'label']
            prot_dir: Directory for protein features
            mol_dir: Directory for molecule features
            model: The trained ContrastiveModel
            device: 'cpu' or 'cuda'
            batch_size: Batch size for inference
        """
        pos_sims = []
        neg_sims = []

        print(f"[Calibrator] Starting Full Retrieval data collection...")
        
        # 1. Pre-load all unique molecule features to avoid repeated I/O
        unique_mols = val_df["label"].unique()
        mol_feats_dict = {}
        
        print(f"[Data] Pre-loading {len(unique_mols)} molecule features...")
        for m_name in tqdm(unique_mols, desc="Loading Molecules"):
            path = os.path.join(mol_dir, f"{m_name}.pt")
            if os.path.exists(path):
                feat = torch.load(path, weights_only=True)
                if feat.dim() == 1: 
                    feat = feat.unsqueeze(0) # Ensure [1, Dim]
                mol_feats_dict[m_name] = feat.to(device)
        
        # Filter valid molecules and create a mapping
        valid_mol_names = list(mol_feats_dict.keys())
        if not valid_mol_names:
            raise ValueError("No valid molecule features found.")
            
        # Stack all molecules into a single tensor: [N_mols, Mol_Dim]
        mol_matrix = torch.cat([mol_feats_dict[m] for m in valid_mol_names], dim=0)
        n_mols = len(valid_mol_names)
        mol_name_to_idx = {name: i for i, name in enumerate(valid_mol_names)}

        model.eval()
        
        # 2. Iterate through proteins and compute scores against the molecule matrix
        print(f"[Data] Computing similarities for {len(val_df)} interactions...")
        
        with torch.no_grad():
            for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing Proteins"):
                p_id = str(row["id"])
                true_mol_name = str(row["label"])
                
                p_path = os.path.join(prot_dir, f"{p_id}.pt")
                if not os.path.exists(p_path) or true_mol_name not in mol_name_to_idx:
                    continue
                
                # Load protein feature: [1, Prot_Dim]
                p_feat = torch.load(p_path, weights_only=True).to(device)
                if p_feat.dim() == 1: 
                    p_feat = p_feat.unsqueeze(0)
                
                # We need to compare this single protein against 'mol_matrix' (N_mols).
                # Since model.predict usually takes pairs, we batch the operation.
                
                # Store all scores for this protein
                all_scores = []
                
                # Batch processing to avoid OOM
                for i in range(0, n_mols, batch_size):
                    end = min(i + batch_size, n_mols)
                    current_batch_size = end - i
                    
                    # Slice molecule batch: [Batch, Mol_Dim]
                    mol_batch = mol_matrix[i:end]
                    
                    # Expand protein to match batch: [Batch, Prot_Dim]
                    p_batch = p_feat.expand(current_batch_size, -1)
                    
                    # Predict: [Batch] (assuming model returns 1D tensor of scores)
                    sim_batch = model.predict(p_batch, mol_batch)
                    all_scores.append(sim_batch.cpu().numpy())
                
                # Concatenate all scores for this protein -> [N_mols]
                all_scores = np.concatenate(all_scores).flatten()
                
                # 3. Separate Positive and Negative scores
                true_idx = mol_name_to_idx[true_mol_name]
                
                # Positive sample
                pos_sims.append(all_scores[true_idx])
                
                # Negative samples (all other molecules)
                # Create a mask to exclude the true positive index
                mask = np.ones(n_mols, dtype=bool)
                mask[true_idx] = False
                neg_sims.extend(all_scores[mask])

        return pos_sims, neg_sims

    def plot_calibration(self, pos_sims, neg_sims, save_path="calibration_plot.png"):
        """
        Visualize the distributions and the probability curve.
        """
        x_grid = np.linspace(-1, 1, 200)
        prob_curve = self.predict_proba(x_grid)

        plt.figure(figsize=(10, 6))
        
        # Plot Histograms
        plt.hist(pos_sims, bins=50, density=True, alpha=0.5, color='green', label='Positive Pairs')
        plt.hist(neg_sims, bins=50, density=True, alpha=0.5, color='red', label='Negative Pairs')
        
        # Plot KDEs
        plt.plot(x_grid, self.kde_pos(x_grid), color='darkgreen', linestyle='--', label='Pos KDE')
        plt.plot(x_grid, self.kde_neg(x_grid), color='darkred', linestyle='--', label='Neg KDE')

        # Plot Probability Curve (Dual Axis)
        ax2 = plt.gca().twinx()
        ax2.plot(x_grid, prob_curve, color='blue', linewidth=3, label='Probability')
        ax2.set_ylabel('Probability P(Pos|Sim)', color='blue')
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='y', labelcolor='blue')

        plt.title("KDE Calibration Result")
        plt.xlabel("Cosine Similarity")
        
        # Combine legends
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        print(f"[Calibrator] Plot saved to {save_path}")
        plt.close()
        
# ==========================================
# Main Execution
# ==========================================
def main():
    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------
    CONFIG = {
        "csv_path": "train_data_to50_augmented_v2.csv",
        "prot_dir": "protein_data",   # Directory containing 1280-dim .pt files
        "mol_dir":  "molecule_data",  # Directory containing 768-dim .pt files
        "weight_path": "contrastive_kfold_fulltrain_jointacc/final_best_20251114_222642.pth",
        "save_pkl_path": "kde_calibrator.pkl",
        "save_plot_path": "calibration_result_all.png",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 256  # Adjust based on GPU memory
    }

    # -------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------
    print(f"[Model] Loading model from {CONFIG['weight_path']}...")
    
    # Initialize model (Ensure dimensions match your training config)
    model = ContrastiveModel(
        protein_dim=1280, 
        molecule_dim=768, 
        projection_dim=128
    )
    
    try:
        state_dict = torch.load(CONFIG["weight_path"], map_location=CONFIG["device"], weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(CONFIG["device"])
    model.eval() # Important: Disable Dropout/BatchNorm training behavior

    # -------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------
    if not os.path.exists(CONFIG["csv_path"]):
        print(f"Error: CSV file not found at {CONFIG['csv_path']}")
        return

    df = pd.read_csv(CONFIG["csv_path"])
    if not {"id", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain ['id', 'label'] columns")

    # -------------------------------------------------
    # 3. Collect Similarities (Full Retrieval)
    # -------------------------------------------------
    calibrator = KDECalibrator()
    
    pos_sims, neg_sims = calibrator.collect_data_full_retrieval(
        val_df=df,
        prot_dir=CONFIG["prot_dir"],
        mol_dir=CONFIG["mol_dir"],
        model=model,
        device=CONFIG["device"],
        batch_size=CONFIG["batch_size"]
    )

    # -------------------------------------------------
    # 4. Fit and Save
    # -------------------------------------------------
    calibrator.fit(pos_sims, neg_sims)
    calibrator.plot_calibration(pos_sims, neg_sims, save_path=CONFIG["save_plot_path"])
    
    with open(CONFIG["save_pkl_path"], "wb") as f:
        pickle.dump(calibrator, f)
    print(f"[System] Calibrator saved to {CONFIG['save_pkl_path']}")

    # -------------------------------------------------
    # 5. Quick Verification
    # -------------------------------------------------
    print("\n--- Verification ---")
    test_scores = [0.1, 0.5, 0.9]
    for s in test_scores:
        p = calibrator.predict_proba(s).item()
        print(f"Score: {s:.2f} -> Probability: {p:.4f}")

if __name__ == "__main__":
    main()

import sys
import __main__

if not hasattr(__main__, "KDECalibrator"):
    setattr(__main__, "KDECalibrator", KDECalibrator)
    
if not hasattr(__main__, "BandwidthCallable"):
    setattr(__main__, "BandwidthCallable", BandwidthCallable)
