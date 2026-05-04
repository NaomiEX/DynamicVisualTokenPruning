import torch
import torch.nn as nn


class LowRankPruner(nn.Module):
    def __init__(self, dim, rank=64, num_heads=8):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads

        # Store Pruner Predictions for training
        self.last_importance_scores = None

        # Low-rank Q, K Projections (D -> r, not D -> D)
        self.q_proj = nn.Linear(dim, rank, bias=False)
        self.k_proj = nn.Linear(dim, rank, bias=False)

        # Merge heads down to single relevance score per visual token
        self.head_merger = nn.Linear(num_heads, 1)

        # GLU scorer: takes [low-rank visual || relevance] and gates it
        # Input: rank + 1, output: 2 (GLU halves it to 1)
        self.gate_proj = nn.Linear(rank + 1, 2, bias=True)
        self.glu = nn.GLU(dim=-1)

    def get_multi_head_attention(self, x_v, x_t):
        B, L, D = x_t.shape
        N = x_v.shape[1]

        # Project to low-rank space [B, L/N, rank] then split into heads
        q = self.q_proj(x_t).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, head_dim]
        k = self.k_proj(x_v).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]

        # Cross attention in low-rank space: [B, H, L, N]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Max over text tokens per head: [B, H, N] -> [B, N, H]
        head_relevance = scores.max(dim=-2).values.transpose(1, 2)

        # Merge heads to single relevance score: [B, N, 1]
        relevance = self.head_merger(head_relevance)
        return relevance

    def forward(self, x_v, x_t, keep_ratio=0.5):
        B, N, D = x_v.shape

        # 1) Cross-attention relevance in low-rank space
        visual_relevance = self.get_multi_head_attention(x_v, x_t)  # [B, N, 1]

        # 2) Project visual tokens to low-rank space for GLU input
        v_proj = self.k_proj(x_v)  # [B, N, rank] — reuse k_proj, same space

        # 3) GLU scoring: gate low-rank visual features by relevance
        combined = torch.cat([v_proj, visual_relevance], dim=-1)     # [B, N, rank+1]
        importance = self.glu(self.gate_proj(combined)).squeeze(-1)  # [B, N, 1] -> [B, N]
        importance = torch.sigmoid(importance)                        # [B, N] in [0, 1]

        # Store scores for backprop
        self.last_importance_scores = importance

        # Soft masking for training
        if self.training:
            return x_v * importance.unsqueeze(-1), None

        # Hard top-k for inference
        keep_k = max(1, int(N * keep_ratio))
        _, keep_idx = torch.topk(importance, keep_k, dim=1)          # [B, keep_k]
        keep_idx, _ = torch.sort(keep_idx, dim=1)

        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, D)        # [B, keep_k, D]
        pruned_v = torch.gather(x_v, 1, gather_idx)

        return pruned_v, keep_idx