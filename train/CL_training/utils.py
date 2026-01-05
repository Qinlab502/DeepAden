# utils.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from protein_feature import load_protein_features
from molecule_feature import load_molecule_features
from kde_t import KDECalibrator

def load_kde_model(kde_path):
    """
    Load the KDE Calibrator model from a pickle file.
    """
    if not os.path.exists(kde_path):
        raise FileNotFoundError(f"KDE model not found at {kde_path}")
    
    print(f"[Utils] Loading KDE model from {kde_path}...")
    with open(kde_path, 'rb') as f:
        calibrator = pickle.load(f)
    return calibrator

def evaluate_model(model, test_loader, mol_feature_dir, device="cuda"):
    """
    Evaluate model retrieval accuracy using molecule names as labels.
    Each protein embedding is compared against all molecule embeddings.
    """
    model.eval()
    device = torch.device(device)

    # === 1. Build molecule embedding bank ===
    all_mol_embeds = []
    all_mol_names = []

    mol_files = sorted([f for f in os.listdir(mol_feature_dir) if f.endswith(".pt")])

    with torch.no_grad():
        for f in mol_files:
            f_path = os.path.join(mol_feature_dir, f)
            # weights_only=True for safety, assuming standard tensors
            mol_feat = torch.load(f_path, map_location=device, weights_only=True)
            if mol_feat.dim() == 1:
                mol_feat = mol_feat.unsqueeze(0)

            mol_proj = model.molecule_projection(mol_feat)
            mol_proj = F.normalize(mol_proj, p=2, dim=1)
            all_mol_embeds.append(mol_proj.cpu())

            mol_name = os.path.splitext(f)[0]
            all_mol_names.append(mol_name)

    all_mol_embeds = torch.cat(all_mol_embeds, dim=0)

    # === 2. Evaluate on test proteins ===
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            p_feat = batch["protein_feature"].to(device)
            p_proj = model.protein_projection(p_feat)
            p_proj = F.normalize(p_proj, p=2, dim=1)

            true_names = batch["label"]

            sim_matrix = torch.mm(p_proj.cpu(), all_mol_embeds.t())
            top_idx = torch.argmax(sim_matrix, dim=1).tolist()
            pred_names = [all_mol_names[i] for i in top_idx]

            batch_correct = sum(p == t for p, t in zip(pred_names, true_names))
            correct += batch_correct
            total += len(true_names)

    acc = correct / total if total > 0 else 0.0

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
    }

def perform_retrieval(model, protein_ids, molecule_labels=None, 
                     protein_feature_dir="data/protein_data", 
                     molecule_feature_dir="data/molecule_data", 
                     kde_path='kde_results/kde_calibrator.pkl', top_k=3, device="cuda", batch_size=64):
    """
    Retrieve molecules for each protein.
    
    Args:
        kde_path: Path to the pickled KDECalibrator. MUST be provided.
        top_k: Number of top molecules to return.
    """
    if kde_path is None:
        raise ValueError("kde_path must be provided to perform calibrated retrieval.")

    model.to(device)
    model.eval()
    
    # Load KDE model (Mandatory)
    calibrator = load_kde_model(kde_path)
    
    # Load protein features
    protein_features = load_protein_features(protein_ids, protein_feature_dir)
    protein_features = protein_features.to(device)
    
    # Get all molecule labels if not provided
    if molecule_labels is None:
        print("No molecule labels provided, using all molecules in directory...")
        molecule_files = [f for f in os.listdir(molecule_feature_dir) if f.endswith('.pt')]
        molecule_labels = [os.path.splitext(os.path.basename(f))[0] for f in molecule_files]
        print(f"Found {len(molecule_labels)} molecules")
       
    num_molecules = len(molecule_labels)
    num_batches = (num_molecules + batch_size - 1) // batch_size
    all_similarities = []
    
    print(f"Processing {num_molecules} molecules in {num_batches} batches...")
    
    # Get protein projections
    with torch.no_grad():
        protein_proj = model.protein_projection(protein_features)
        protein_proj = F.normalize(protein_proj, p=2, dim=1)
    
    # Process molecules in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_molecules)
        batch_molecules = molecule_labels[start_idx:end_idx]
        
        batch_features = load_molecule_features(batch_molecules, molecule_feature_dir)
        batch_features = batch_features.to(device)
        
        with torch.no_grad():
            molecule_proj = model.molecule_projection(batch_features)
            molecule_proj = F.normalize(molecule_proj, p=2, dim=1)
            
            batch_similarity = torch.mm(protein_proj, molecule_proj.t())
            all_similarities.append(batch_similarity.cpu())
    
    # Concatenate all similarity matrices
    if len(all_similarities) > 1:
        similarity = torch.cat(all_similarities, dim=1).numpy()
    else:
        similarity = all_similarities[0].numpy()
    
    # For each protein, process molecules
    results = []
    for i, protein_id in enumerate(protein_ids):
        sim_scores = similarity[i]
        
        # Sort by similarity descending (High Cosine Similarity -> High Probability)
        # We sort by raw similarity first because it's faster and monotonic with probability
        sorted_indices = np.argsort(-sim_scores)
        
        # Select Top-K
        top_indices = sorted_indices[:top_k]
        top_molecules = [molecule_labels[int(idx)] for idx in top_indices]
        top_sim_scores = [sim_scores[int(idx)] for idx in top_indices]
        
        # Calculate probabilities for the Top-K results using the calibrator
        # predict_proba expects a numpy array
        probs = calibrator.predict_proba(np.array(top_sim_scores))
        
        result = {
            'protein_id': protein_id,
            'molecules': top_molecules,
            'probabilities': probs.tolist()      # Calibrated Probability
        }
        
        results.append(result)
    
    return results

def save_results(results, output_file="retrieval_results.csv"):
    """
    Save retrieval results. 
    It now dynamically saves however many molecules are in the results (determined by top_k).
    """
    rows = []
    
    for result in results:
        protein_id = result['protein_id']
        molecules = result['molecules']
        scores = result['probabilities']
        
        row = {'protein_id': protein_id}
        
        # Use the actual length of the results list
        n_molecules = len(molecules)
        
        for i in range(n_molecules):
            rank = i + 1
            row[f'Top{rank}'] = molecules[i]             
            row[f'Top{rank}_score'] = round(scores[i], 2)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    file_ext = os.path.splitext(output_file)[1].lower()
    if file_ext == '.xlsx':
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")