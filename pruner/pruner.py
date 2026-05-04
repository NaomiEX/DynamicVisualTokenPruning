import torch
import torch.nn as nn


class CLIPAttentionPruner(nn.Module):
    # Baseline: scores tokens by how much CLIP's CLS token attends to each patch
    # in the last vision encoder layer. _clip_attentions must be set before each forward call (done by PrunableLlava).
    # Same interface as QueryAwarePruner: forward(x_v, x_t, keep_ratio) -> (pruned, idx)
    

    def __init__(self):
        super().__init__()
        self.last_importance_scores = None
        self._clip_attentions = None  # set externally before forward

    def forward(self, x_v, x_t, keep_ratio=0.5):
        # x_v: [B, N, D] projected patch tokens (CLS already excluded)
        # x_t: unused, here for interface compat
        assert self._clip_attentions is not None, "_clip_attentions not set"

        # last layer CLS->patch attention, averaged over heads
        last_attn = self._clip_attentions[-1]        # [B, H, S, S] where S = N+1
        cls_attn = last_attn[:, :, 0, 1:]            # [B, H, N]
        importance = cls_attn.float().mean(dim=1)     # [B, N]

        # normalize to [0, 1]
        lo = importance.min(dim=-1, keepdim=True).values
        hi = importance.max(dim=-1, keepdim=True).values
        importance = (importance - lo) / (hi - lo + 1e-8)

        self.last_importance_scores = importance

        if self.training:
            return x_v * importance.unsqueeze(-1), None

        # hard top-k selection
        keep_k = max(1, int(x_v.shape[1] * keep_ratio))
        _, keep_idx = torch.topk(importance, keep_k, dim=1)
        keep_idx, _ = torch.sort(keep_idx, dim=1)

        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, x_v.shape[-1])
        pruned_v = torch.gather(x_v, 1, gather_idx)

        return pruned_v, keep_idx


class LLMAttentionPruner(nn.Module):
    # Baseline: run first K layers of the LLM on the full (unpruned) merged sequence,
    # then score image tokens by how much text tokens attend to them.
    # Query-aware (unlike CLIPAttentionPruner) but costs a partial LLM forward pass.
    #
    # _llm_attentions and _image_mask must be set before forward (done by PrunableLlava).
    # Same interface as QueryAwarePruner.

    def __init__(self, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.last_importance_scores = None
        self._llm_attentions = None  # list of [B, H, seq, seq] from first K layers
        self._image_mask = None      # [seq] bool mask, True for image token positions

    def forward(self, x_v, x_t, keep_ratio=0.5):
        # x_v: [B, N, D] image features (not used for scoring, just for pruning)
        # x_t: unused
        assert self._llm_attentions is not None, "_llm_attentions not set"
        assert self._image_mask is not None, "_image_mask not set"

        image_mask = self._image_mask  # [seq]
        image_positions = torch.where(image_mask)[0]  # indices of image tokens
        text_positions = torch.where(~image_mask)[0]   # indices of text tokens

        # average attention across layers and heads
        # each is [B, H, seq, seq] -> stack -> [B, K, H, seq, seq] -> mean over K,H
        attn_stack = torch.stack(self._llm_attentions, dim=1)  # [B, K, H, seq, seq]
        attn_avg = attn_stack.float().mean(dim=(1, 2))         # [B, seq, seq]

        # how much text tokens attend to each image token
        # attn_avg[:, text_positions, :][:, :, image_positions] -> [B, n_text, N]
        text_to_image = attn_avg[:, text_positions][:, :, image_positions]  # [B, n_text, N]
        importance = text_to_image.mean(dim=1)  # [B, N] avg over text positions

        # normalize to [0, 1]
        lo = importance.min(dim=-1, keepdim=True).values
        hi = importance.max(dim=-1, keepdim=True).values
        importance = (importance - lo) / (hi - lo + 1e-8)

        self.last_importance_scores = importance

        if self.training:
            return x_v * importance.unsqueeze(-1), None

        # hard top-k
        keep_k = max(1, int(x_v.shape[1] * keep_ratio))
        _, keep_idx = torch.topk(importance, keep_k, dim=1)
        keep_idx, _ = torch.sort(keep_idx, dim=1)

        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, x_v.shape[-1])
        pruned_v = torch.gather(x_v, 1, gather_idx)

        return pruned_v, keep_idx


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