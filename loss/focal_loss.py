import math
import torch
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # logits (B, N)
        p = F.softmax(logits, 1)
        p_t = torch.gather(p, 1, labels[:, None])
        ce = F.cross_entropy(logits, labels, reduction='none')
        loss = ce * ((1 - p_t) ** self.gamma)
        return loss.mean()