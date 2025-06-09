# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, protein_embeds, molecule_embeds, labels=None):
        # Normalize embeddings
        protein_embeds = F.normalize(protein_embeds, p=2, dim=1)
        molecule_embeds = F.normalize(molecule_embeds, p=2, dim=1)
        
        # Calculate similarity scores
        similarity = torch.mm(protein_embeds, molecule_embeds.t()) / self.temperature
        
        if labels is not None:
            # Supervised contrastive loss with labels
            # Compute the cross entropy loss for each pair using the labels
            batch_size = protein_embeds.size(0)
            labels = labels.view(-1, 1)  # Shape: [batch_size, 1]
            
            # Calculate direct similarity between paired protein and molecule
            diagonal = torch.diag(similarity)  # Shape: [batch_size]
            
            # Use BCE loss with the raw similarity scores
            loss = F.binary_cross_entropy_with_logits(
                diagonal, 
                labels.squeeze(),
                reduction='mean'
            )
            
            return loss
        else:
            # Unsupervised contrastive loss (assuming diagonal is positive pairs)
            pos_sim = torch.diag(similarity)
            
            # For each protein, all other molecules are negatives
            exp_sim = torch.exp(similarity)
            
            # Mask out the self-similarity
            mask = torch.ones_like(similarity)
            mask.fill_diagonal_(0)
            
            neg_sim = torch.sum(exp_sim * mask, dim=1)
            
            # InfoNCE loss
            loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim + 1e-8))
            return loss.mean()