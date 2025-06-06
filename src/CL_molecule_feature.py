# molecule_feature.py
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def ensure_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def precompute_molecule_features(csv_file, device, output_dir='data/molecule_data', batch_size=128):
    ensure_dirs(output_dir)
    
    print("Loading MoLFormer model...")
    mol_model_path = "model/MoLFormer-XL-both-10pct"
    mol_tokenizer = AutoTokenizer.from_pretrained(mol_model_path, trust_remote_code=True)
    mol_model = AutoModel.from_pretrained(mol_model_path, deterministic_eval=True, trust_remote_code=True)
    mol_model.eval()
    mol_model.to(device)
    
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if 'SMILES' not in df.columns or 'labels' not in df.columns:
        raise ValueError("CSV file must contain both 'SMILES' and 'labels' columns")
    
    label_smiles_pairs = df[['labels', 'SMILES']].drop_duplicates().values.tolist()
    print(f"Found {len(df)} entries and {len(label_smiles_pairs)} unique label-SMILES pairs")
    
    with tqdm(total=len(label_smiles_pairs), desc="Processing Molecules", unit="mol") as pbar:
        for start_idx in range(0, len(label_smiles_pairs), batch_size):
            end_idx = min(start_idx + batch_size, len(label_smiles_pairs))
            batch_pairs = label_smiles_pairs[start_idx:end_idx]
            batch_labels = [pair[0] for pair in batch_pairs]
            batch_smiles = [pair[1] for pair in batch_pairs]
            
            try:
                inputs = mol_tokenizer(batch_smiles, padding=True, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                
                with torch.no_grad():
                    outputs = mol_model(**inputs)
                    mol_embeddings = outputs.pooler_output
                
                for idx, label in enumerate(batch_labels):
                    out_file = os.path.join(output_dir, f"{label}.pt")
                    torch.save(mol_embeddings[idx].cpu(), out_file)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                for label, smiles in zip(batch_labels, batch_smiles):
                    try:
                        inputs = mol_tokenizer([smiles], padding=True, return_tensors="pt")
                        inputs = {key: value.to(device) for key, value in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = mol_model(**inputs)
                            mol_embedding = outputs.pooler_output[0]
                        
                        out_file = os.path.join(output_dir, f"{label}.pt")
                        torch.save(mol_embedding.cpu(), out_file)
                        
                    except Exception as e2:
                        print(f"Error processing individual molecule: {label}, SMILES: {smiles}, Error: {e2}")
                        with open(os.path.join(output_dir, "failed_molecules.txt"), "a") as f:
                            f.write(f"{label}\t{smiles}\n")
            
            pbar.update(len(batch_pairs))
    
    print(f"All molecule features have been processed and saved to {output_dir}")

def load_molecule_features(labels_list, feature_dir="data/molecule_data"):
    features = []
    missing_labels = []
    
    for label in labels_list:
        feature_path = os.path.join(feature_dir, f"{label}.pt")
        if os.path.exists(feature_path):
            feature = torch.load(feature_path)
            features.append(feature)
        else:
            missing_labels.append(label)
            print(f"Warning: No feature file found for label: {label}")
            
            if features:
                zero_feature = torch.zeros_like(features[0])
            else:
                zero_feature = torch.zeros(1024)
            features.append(zero_feature)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} labels not found in precomputed features.")
        if len(missing_labels) < 10:
            print(f"Missing labels: {missing_labels}")
        else:
            print(f"First 10 missing labels: {missing_labels[:10]}...")
    
    return torch.stack(features)

if __name__ == "__main__":
    precompute_molecule_features(
        csv_file="./data/train_set.csv",
        output_dir="./data/molecule_data"
    )