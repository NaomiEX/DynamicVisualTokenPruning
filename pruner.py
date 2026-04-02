import torch
import torch.nn as nn

class QueryAwarePruner(nn.Module):
    def __init__(self, dim, num_heads=8, use_multi_head=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_multi_head = use_multi_head
        self.head_dim = dim // num_heads
        
        # Store Pruner Predictions for training
        self.last_importance_scores = None 
        
        # Q, K Projections 
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        
        if use_multi_head:
            self.head_merger = nn.Linear(num_heads, 1)
        
        # Final MLP Scorer
        self.mlp_scorer = nn.Sequential(
            nn.Linear(dim + 1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def get_normal_attention(self, x_v, x_t):
        Q = self.q_proj(x_t) 
        K = self.k_proj(x_v)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.dim ** 0.5)
        relevance = scores.max(dim=1).values.unsqueeze(-1)
        return relevance

    def get_multi_head_attention(self, x_v, x_t):
        B, L, D = x_t.shape
        N = x_v.shape[1]
        q = self.q_proj(x_t).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_v).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        head_relevance = scores.max(dim=-2).values.transpose(1, 2)
        relevance = self.head_merger(head_relevance)
        return relevance

    def forward(self, x_v, x_t, keep_ratio=0.5):
        # Calculate Relevance score
        if self.use_multi_head:
            visual_relevance = self.get_multi_head_attention(x_v, x_t)
        else:
            visual_relevance = self.get_normal_attention(x_v, x_t)
            
        # MLP Forward Pass
        combined = torch.cat([x_v, visual_relevance], dim=-1)
        importance = self.mlp_scorer(combined).squeeze(-1) # [B, N, 1] -> [B, N]
        
        # Store the scores for backprop
        self.last_importance_scores = importance 
        
        # Use Soft Masking to allow gradients to flow for Training
        if self.training:
            return x_v * importance.unsqueeze(-1), None
    
        # Inference (Hard Pruning)
        keep_k = max(1, int(x_v.shape[1] * keep_ratio))
        _, keep_idx = torch.topk(importance, keep_k, dim=1) # [B, keep_k]
        keep_idx, _ = torch.sort(keep_idx, dim=1)
        
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, self.dim) # [B, keep_k, 4096]
        # print("Gather Indices Shape:", gather_idx.shape) # Debugging line
        pruned_v = torch.gather(x_v, 1, gather_idx)
        
        return pruned_v, keep_idx