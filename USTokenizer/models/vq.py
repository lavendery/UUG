from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.functional as F
from .layers import WNConv1d

class VectorQuantize(nn.Module):
    """
    a vector quantization layer, adopted from DAC-Codec
    """
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z):
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        z_q = z_e + (z_q - z_e).detach()  # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = self.out_proj(z_q)
        return z_q, indices, commitment_loss, codebook_loss

    def embed_code(self, embed_id):
        codebook = self.codebook.weight 
        return F.embedding(embed_id, codebook)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        # L2 normalize encodings and codebook (ViT-VQGAN)
        codebook = self.codebook.weight 
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices
