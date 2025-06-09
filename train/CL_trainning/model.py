# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(ProjectionHead, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class ContrastiveModel(nn.Module):
    def __init__(self, protein_dim=256, molecule_dim=768, projection_dim=128):
        super(ContrastiveModel, self).__init__()
        
        # Projection heads to bring both embeddings to the same space
        self.protein_projection = ProjectionHead(protein_dim, projection_dim)
        self.molecule_projection = ProjectionHead(molecule_dim, projection_dim)
        
    def forward(self, protein_features, molecule_features):
        protein_proj = self.protein_projection(protein_features)
        molecule_proj = self.molecule_projection(molecule_features)
        
        return protein_proj, molecule_proj
    
    def predict(self, protein_features, molecule_features):
        protein_proj = self.protein_projection(protein_features)
        molecule_proj = self.molecule_projection(molecule_features)
        
        # Normalize embeddings
        protein_proj = F.normalize(protein_proj, p=2, dim=1)
        molecule_proj = F.normalize(molecule_proj, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.sum(protein_proj * molecule_proj, dim=1)
        
        return similarity