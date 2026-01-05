# protein_feature.py
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM
from peft import PeftModel

def ensure_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def precompute_protein_features(csv_file, output_dir='../data/protein_data', 
                               device="cuda", batch_size=256):
    """
    Precompute protein features using ESM2 model.
    The output embeddings will be 1280-dimensional vectors.
    """
    ensure_dirs(output_dir)
    
    print("Loading ESM model...")
    model_name = '../pretrained_model/esm_model/esm2_t33_650M_UR50D'
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    base_model = EsmForMaskedLM.from_pretrained(model_name)
    esm_model = PeftModel.from_pretrained(base_model, '../pretrained_model/esm_model/lora_esm2_650M/')
    esm_model.eval()
    esm_model.to(device)
    
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    ids = df['id'].tolist()
    seqs = df['pocket'].tolist()
    
    print(f"Found {len(ids)} sequences.")
    print(f"ESM2 embedding dimension: 1280")
    
    with tqdm(total=len(seqs), desc="Processing Sequences", unit="seq") as pbar:
        for start_idx in range(0, len(seqs), batch_size):
            end_idx = min(start_idx + batch_size, len(seqs))
            batch_seqs = seqs[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                # Get ESM2 embeddings (last hidden state)
                esm_output = esm_model.esm(**inputs).last_hidden_state
                
                for idx, seq_id in enumerate(batch_ids):
                    # Get sequence length (excluding special tokens)
                    seq_length = len(batch_seqs[idx]) + 2  # +2 for [CLS] and [SEP]
                    
                    # Extract sequence embedding (mean pooling over sequence length, excluding special tokens)
                    seq_embedding = esm_output[idx, 1:seq_length-1, :].mean(0)
                    
                    # Save the raw ESM2 embedding (1280-dim)
                    out_file = os.path.join(output_dir, f"{seq_id}.pt")
                    torch.save(seq_embedding.cpu(), out_file)
            
            pbar.update(len(batch_seqs))
    
    print(f"All protein features have been processed and saved to {output_dir}")
    print(f"Each feature file contains a 1280-dimensional embedding vector")

def load_protein_features(protein_ids, feature_dir="../data/protein_data"):
    """
    Load precomputed protein features.
    
    Args:
        protein_ids: List of protein IDs
        feature_dir: Directory containing the precomputed features
    
    Returns:
        Tensor of shape (len(protein_ids), 1280)
    """
    features = []
    for protein_id in protein_ids:
        feature_path = os.path.join(feature_dir, f"{protein_id}.pt")
        if os.path.exists(feature_path):
            feature = torch.load(feature_path)
            features.append(feature)
        else:
            raise FileNotFoundError(f"Feature file not found for protein {protein_id}")
    
    return torch.stack(features)

if __name__ == "__main__":
    precompute_protein_features(
        csv_file="train_data_to50_augmented_v2.csv",  # Path to your CSV file with 'id' and 'pocket' columns
        output_dir="./protein_data",
        device="cuda",
        batch_size=256
    )