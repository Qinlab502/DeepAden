# protein_feature.py
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM
from peft import PeftModel

def ensure_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, hidden_dim, out_dim, device, drop_out=0.1):
        dtype = torch.float32
        model = cls(hidden_dim, out_dim, device, dtype, drop_out)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model

def precompute_protein_features(csv_file, output_dir='results/protein_data', 
                               hidden_dim=512, out_dim=256, device="cuda", drop_out=0.1, batch_size=256):
    ensure_dirs(output_dir)
    
#     print("Loading ESM model...")
    model_name = 'model/esm2_t33_650M_UR50D'
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    base_model = EsmForMaskedLM.from_pretrained(model_name)
    esm_model = PeftModel.from_pretrained(base_model, 'model/lora_esm2_650M/')
    esm_model.eval()
    esm_model.to(device)
    
#     print("Loading LayerNormNet model...")
    layernorm_checkpoint = "model/data_dup_supconH_0331.pth"
    layernorm_model = LayerNormNet.load_from_checkpoint(
        layernorm_checkpoint, 
        hidden_dim, 
        out_dim, 
        device, 
        drop_out
    )
    
#     print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    ids = df['id'].tolist()
    seqs = df['pocket'].tolist()
    
#     print(f"Found {len(ids)} sequences.")
    
    with tqdm(total=len(seqs), desc="Processing Sequences", unit="seq") as pbar:
        for start_idx in range(0, len(seqs), batch_size):
            end_idx = min(start_idx + batch_size, len(seqs))
            batch_seqs = seqs[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                esm_output = esm_model.esm(**inputs).last_hidden_state
                
                for idx, seq_id in enumerate(batch_ids):
                    seq_length = len(batch_seqs[idx]) + 2  
                    seq_embedding = esm_output[idx, 1:seq_length-1, :].mean(0)
                    
                    processed_embedding = layernorm_model(seq_embedding.unsqueeze(0)).squeeze(0)
                    
                    out_file = os.path.join(output_dir, f"{seq_id}.pt")
                    torch.save(processed_embedding.cpu(), out_file)
            
            pbar.update(len(batch_seqs))
    
    print(f"All protein features have been processed and saved to {output_dir}")

def load_protein_features(protein_ids, feature_dir="results/protein_data"):
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
        csv_file="./data/train_set.csv",  # Path to your CSV file with 'id' and 'seq' columns
        output_dir="./data/protein_data",
        hidden_dim=512,
        out_dim=256
    )