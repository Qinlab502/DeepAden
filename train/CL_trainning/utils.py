# utils.py
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from protein_feature import load_protein_features
from molecule_feature import load_molecule_features

def maximum_separation(dist_lst, first_grad=True, use_max_grad=True):
    """Find the optimal separation point in a sorted list of similarity values"""
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1] - sep_lst[1:])
    
    if use_max_grad:
        # Use the point with maximum gradient
        max_sep_i = np.argmax(sep_grad)
    else:
        # Use first or last large gradient
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    
    # Default to index 2 if no suitable separation point found
    if max_sep_i >= 3:
        max_sep_i = 2
    
    return max_sep_i

def infer_confidence_gmm(distance, gmm_lst):
    """Calculate confidence score for a prediction using ensemble of GMM models"""
    confidence = []
    for j in range(len(gmm_lst)):
        main_GMM = gmm_lst[j]
        a, b = main_GMM.means_
        # Identify which component corresponds to the positive class
        true_model_index = 1 if a[0] < b[0] else 0
        certainty = main_GMM.predict_proba([[distance]])[0][true_model_index]
        confidence.append(certainty)
    # Return average confidence across all GMM models
    return np.mean(confidence)

def evaluate_model(model, test_loader, device="cuda"):
    """Evaluate model performance on test data and return metrics"""
    model.eval()
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            protein_features = batch['protein_feature'].to(device)
            molecule_features = batch['molecule_feature'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Get projections
            protein_proj, molecule_proj = model(protein_features, molecule_features)
            
            # Normalize embeddings
            protein_proj = F.normalize(protein_proj, p=2, dim=1)
            molecule_proj = F.normalize(molecule_proj, p=2, dim=1)
            
            # Calculate cosine similarity
            similarity = torch.sum(protein_proj * molecule_proj, dim=1).cpu().numpy()
            
            all_similarities.extend(similarity)
            all_labels.extend(labels)
    
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(all_labels, all_similarities)
    auprc = average_precision_score(all_labels, all_similarities)
    
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    return {'auroc': auroc, 'auprc': auprc, 'similarities': all_similarities, 'labels': all_labels}

def load_gmm_models(gmm_dir):
    """Load GMM models from pickle files in the specified directory"""
    import pickle
    
    gmm_models = []
    for filename in os.listdir(gmm_dir):
        if filename.endswith(".pkl") and filename.startswith("GMM_model_exp_"):
            filepath = os.path.join(gmm_dir, filename)
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                gmm_models.append(model_data['gmm_model'])
    
    print(f"Loaded {len(gmm_models)} GMM models")
    return gmm_models

def perform_retrieval(model, protein_ids, molecule_labels=None, 
                     protein_feature_dir="data/protein_data", 
                     molecule_feature_dir="data/molecule_data", 
                     gmm_dir=None, top_k=10, device="cuda", batch_size=64):
    """
    Retrieve molecules for each protein
    
    Two modes:
    1. Without GMM (gmm_dir=None): Return top-k molecules with similarity scores
    2. With GMM: Use maximum_separation to find significant point and return confidence scores 
                for molecules up to that point
    """
    model.to(device)
    model.eval()
    
    # Load GMM models if provided
    gmm_models = None
    if gmm_dir is not None:
        gmm_models = load_gmm_models(gmm_dir)
    
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
        
        if gmm_models is not None:
            # GMM mode - use maximum_separation to find significant point
            sorted_indices = np.argsort(-sim_scores)
            sorted_scores = sim_scores[sorted_indices]
            
            max_sep_idx = maximum_separation(sorted_scores, first_grad=True, use_max_grad=True)
#             print(f"Protein {protein_id}: Significant point at position {max_sep_idx}")
            
            # Collect molecules up to the significant point
            significant_indices = sorted_indices[:max_sep_idx+1]
            significant_molecules = [molecule_labels[idx] for idx in significant_indices]
            significant_scores = [sim_scores[idx] for idx in significant_indices]
            
            # Calculate confidence scores for significant molecules
            confidence_scores = [infer_confidence_gmm(score, gmm_models) for score in significant_scores]
            
            result = {
                'protein_id': protein_id,
                'molecules': significant_molecules,
                'confidence_scores': confidence_scores
            }
            
        else:
            # No GMM - return top-k with similarity scores
            top_indices = np.argsort(-sim_scores)[:top_k]
            top_molecules = [molecule_labels[int(idx)] for idx in top_indices]
            top_scores = [sim_scores[int(idx)] for idx in top_indices]
            
            result = {
                'protein_id': protein_id,
                'molecules': top_molecules,
                'similarity_scores': top_scores
            }
        
        results.append(result)
    
    return results

def save_results(results, output_file="retrieval_results.csv", max_molecules=10):
    """
    Save retrieval results with each protein in one row
    
    Args:
        results: output from retrieval function
        output_file: output file path (.csv or .xlsx)
        max_molecules: maximum number of molecules per protein
    """
    import pandas as pd
    import os
    
    rows = []
    
    for result in results:
        protein_id = result['protein_id']
        molecules = result['molecules']
        
        # Determine score type
        if 'confidence_scores' in result:
            scores = result['confidence_scores']
            score_type = 'confidence'
        else:
            scores = result['similarity_scores'] 
            score_type = 'probability'
        
        row = {'protein_id': protein_id}
        
        n_molecules = min(len(molecules), max_molecules)
        
        for i in range(n_molecules):
            row[f'molecule_{i+1}'] = molecules[i]             
            row[f'{score_type}_score_{i+1}'] = round(scores[i], 2)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    file_ext = os.path.splitext(output_file)[1].lower()
    if file_ext == '.xlsx':
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    
