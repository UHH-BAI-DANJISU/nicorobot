import torch
import torch.nn.functional as F

def info_nce_loss(e_pos, e_neg):
    """
    e_pos: [B,1]
    e_neg: [B,K]
    """
    logits = torch.cat([-e_pos, -e_neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
