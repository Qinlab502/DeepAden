import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss with optional bidirectional formulation.

    - If `labels` provided:
        (i, j) pairs sharing the same label are treated as positives.
        Includes self-pairs: (protein_i, molecule_i) is always positive.
        Computes bidirectional contrastive objective:
            protein→molecule  and  molecule→protein
    - If no labels provided:
        Fallback to unsupervised InfoNCE (positive = diagonal only).
    """

    def __init__(self, temperature=0.1, contrastive_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

    def forward(self, protein_embeds, molecule_embeds, labels=None):
        """
        Args:
            protein_embeds: [B, D]
            molecule_embeds: [B, D]
            labels: Optional [B], int or long.
        Returns:
            Scalar loss.
        """
        # Normalize embeddings
        protein_embeds = F.normalize(protein_embeds, p=2, dim=1)
        molecule_embeds = F.normalize(molecule_embeds, p=2, dim=1)

        # Similarity matrix: [B, B]
        logits = torch.mm(protein_embeds, molecule_embeds.t()) / self.temperature
        B = logits.size(0)
        eps = 1e-8

        # ============================================================
        # Case 1: Supervised Contrastive Learning (label-based)
        # ============================================================
        if labels is not None:
            labels = labels.view(-1, 1)
            # same label → positive (includes diagonal)
            label_mask = (labels == labels.T).float()  # [B, B]
            
            # count positives for each anchor
            num_positives = label_mask.sum(dim=1)  # [B]

            # === Protein → Molecule ===
            exp_logits = torch.exp(logits)  # [B, B]
            # denominator: all molecules
            denom_p2m = exp_logits.sum(dim=1, keepdim=True)  # [B, 1]
            # log probability for all pairs
            log_prob_p2m = logits - torch.log(denom_p2m + eps)  # [B, B]
            # average over all positives
            loss_p2m = -(label_mask * log_prob_p2m).sum(dim=1) / (num_positives + eps)

            # === Molecule → Protein === (symmetric direction)
            exp_logits_T = exp_logits.T  # [B, B]
            # denominator: all proteins
            denom_m2p = exp_logits_T.sum(dim=1, keepdim=True)  # [B, 1]
            # log probability for all pairs
            log_prob_m2p = logits.T - torch.log(denom_m2p + eps)  # [B, B]
            # average over all positives
            loss_m2p = -(label_mask.T * log_prob_m2p).sum(dim=1) / (num_positives + eps)
            
            # Combine bi-directional losses
            supervised_contrastive = 0.5 * (loss_p2m.mean() + loss_m2p.mean())

            return self.contrastive_weight * supervised_contrastive

        # ============================================================
        # Case 2: Unsupervised (InfoNCE)
        # ============================================================
        else:
            pos_sim = torch.diag(logits)
            exp_sim = torch.exp(logits)

            # Mask out self for negatives
            mask = torch.ones_like(logits)
            mask.fill_diagonal_(0)

            # denominator = pos + neg
            neg_sum = torch.sum(exp_sim * mask, dim=1)
            unsupervised_contrastive = -torch.log(
                torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sum + eps)
            )
            unsupervised_contrastive = unsupervised_contrastive.mean()

            return self.contrastive_weight * unsupervised_contrastive