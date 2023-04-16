import math
import torch
from torch.nn import functional as F

class ArcFace(torch.nn.Module):
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""

    def __init__(self, num_classes, embedding_size=512, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        centers = torch.zeros(num_classes, embedding_size)
        torch.nn.init.xavier_normal_(centers)
        self.centers = torch.nn.Parameter(centers)
        self.scale = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)*margin

    def forward(self, embedding_features: torch.Tensor, labels: torch.Tensor, return_cosine=False):
        """Calculate ArcFace loss

        Parameters
        ----------
        embedding_features : torch.Tensor
            Backbone's output with 
            shape (B, D) with D is the embedding size
        labels : torch.Tensor
            Shape (B) 
            
        Returns
        -------
        loss: torch.Tensor
        """
        norm_emb_ft = F.normalize(embedding_features)
        norm_centers = F.normalize(self.centers)
        one_hot = F.one_hot(labels, num_classes=self.num_classes)
        
        cos_th = F.linear(norm_emb_ft, norm_centers).clamp(-1, 1)
        sin_th = (1 - cos_th ** 2).sqrt()
        
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        margin_mask = (cos_th > self.th).to(cos_th.dtype)
        cos_th_m = margin_mask * cos_th_m + (1 - margin_mask) * (cos_th - self.mm)
        
        logits = (one_hot * cos_th_m + (1 - one_hot) * cos_th) * self.scale
        loss = F.cross_entropy(logits, labels)
        if return_cosine:
            return loss, cos_th
        return loss
        
    
    
# for debug
if __name__ == "__main__":
    num_classes = 100
    embedding_size = 512
    batch_size = 64
    af = ArcFace(num_classes)
    emb = torch.rand(batch_size, embedding_size)
    labels = torch.randint(0, num_classes, size=(batch_size,))
    out = af(emb, labels)
    print(out)
        