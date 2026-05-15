"""FastV (Chen et al., ECCV 2024) for Qwen2.5-VL.

Decoder-internal, training-free image-token pruning. At layer K-1 we score
image tokens by the attention they receive from the last prompt token,
drop the bottom R fraction, and run layers K..N-1 on the reduced sequence.
KV cache lengths therefore differ across layers from K onward.

Enable: instantiate FastVQwen2_5_VLForConditionalGeneration in place of
Qwen2_5_VLForConditionalGeneration. K and R are constructor args
(fastv_k, fastv_r); defaults K=2, R=0.5. fastv_r=0.0 disables.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    apply_multimodal_rotary_pos_emb,
)


class FastVQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, fastv_k: int = 2, fastv_r: float = 0.5):
        super().__init__(config)
        self.fastv_k = fastv_k
        self.fastv_r = fastv_r
        self.last_pruning_stats: Optional[dict] = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        logits_to_keep=0,
        **kwargs,
    ):
        is_prefill = past_key_values is None or (
            hasattr(past_key_values, "get_seq_length")
            and past_key_values.get_seq_length() == 0
        )
        do_prune = (
            is_prefill
            and pixel_values is not None
            and input_ids is not None
            and image_grid_thw is not None
            and self.fastv_k > 0
            and 0.0 < self.fastv_r < 1.0
        )

        if not do_prune:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if labels is not None:
            raise NotImplementedError("FastV is inference-only; label remapping not implemented.")

        text_model = self.model.language_model
        text_config = text_model.config
        device = input_ids.device

        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        if hasattr(image_embeds, "pooler_output"):
            image_embeds = image_embeds.pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
            if hasattr(video_embeds, "pooler_output"):
                video_embeds = video_embeds.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == self.config.image_token_id] = 1
        video_tok = getattr(self.config, "video_token_id", None)
        if video_tok is not None:
            mm_token_type_ids[input_ids == video_tok] = 2
        position_ids_full, rope_deltas_full = self.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )
        self.model.rope_deltas = rope_deltas_full

        if use_cache is None:
            use_cache = text_config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=text_config)

        seq_len = inputs_embeds.shape[1]
        cache_position_full = torch.arange(seq_len, device=device)

        mask_kwargs_full = {
            "config": text_config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position_full,
            "past_key_values": past_key_values,
            "position_ids": None,
        }
        masks_full = {"full_attention": create_causal_mask(**mask_kwargs_full)}
        if text_model.has_sliding_layers:
            masks_full["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs_full)

        position_embeddings_full = text_model.rotary_emb(inputs_embeds, position_ids_full)

        K = min(self.fastv_k, len(text_model.layers))
        hidden_states = inputs_embeds

        layer_types = text_model.config.layer_types
        for i, layer in enumerate(text_model.layers[: K - 1]):
            layer_out = layer(
                hidden_states,
                attention_mask=masks_full[layer_types[i]],
                position_ids=None,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position_full,
                position_embeddings=position_embeddings_full,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        layer_k = text_model.layers[K - 1]
        image_token_id = self.config.image_token_id
        image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        n_img = int(image_positions.numel())

        attn = layer_k.self_attn
        normed = layer_k.input_layernorm(hidden_states)
        bsz = normed.shape[0]
        q = attn.q_proj(normed).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(normed).view(bsz, seq_len, -1, attn.head_dim).transpose(1, 2)
        cos, sin = position_embeddings_full
        mrope_section = text_config.rope_scaling["mrope_section"]
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        num_kv_groups = q.shape[1] // k.shape[1]
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
        q_last = q[:, :, -1:, :]
        score_row = (q_last @ k.transpose(-2, -1)) * attn.scaling
        score_row = score_row.float().softmax(dim=-1)
        scores = score_row[0, :, 0, image_positions].mean(dim=0)

        layer_out = layer_k(
            hidden_states,
            attention_mask=masks_full[layer_types[K - 1]],
            position_ids=None,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position_full,
            position_embeddings=position_embeddings_full,
        )
        hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        keep_n = max(1, int(round(n_img * (1.0 - self.fastv_r))))
        _, top_local = torch.topk(scores, keep_n)
        keep_image_positions = image_positions[top_local]

        keep_seq = torch.ones(seq_len, dtype=torch.bool, device=device)
        keep_seq[image_positions] = False
        keep_seq[keep_image_positions] = True
        keep_idx = keep_seq.nonzero(as_tuple=True)[0]

        self.last_pruning_stats = {"original_n": n_img, "pruned_n": keep_n}

        hidden_states = hidden_states.index_select(1, keep_idx)
        cos, sin = position_embeddings_full
        position_embeddings_pruned = (
            cos.index_select(-2, keep_idx),
            sin.index_select(-2, keep_idx),
        )
        # After pruning, the post-K layers operate on S_pruned tokens. The
        # transformers 5.x mask builder reads kv-length from past_key_values,
        # which was just populated to length S by layers[:K]. Hide it for the
        # mask construction; the layer.forward calls still use the real cache.
        cache_position_pruned = torch.arange(len(keep_idx), device=device)
        attn_pruned = (
            attention_mask.index_select(1, keep_idx) if attention_mask is not None else None
        )
        mask_kwargs_pruned = {
            "config": text_config,
            "input_embeds": hidden_states,
            "attention_mask": attn_pruned,
            "cache_position": cache_position_pruned,
            "past_key_values": None,
            "position_ids": None,
        }
        masks_pruned = {"full_attention": create_causal_mask(**mask_kwargs_pruned)}
        if text_model.has_sliding_layers:
            masks_pruned["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs_pruned)

        for i, layer in enumerate(text_model.layers[K:], start=K):
            layer_out = layer(
                hidden_states,
                attention_mask=masks_pruned[layer_types[i]],
                position_ids=None,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position_pruned,
                position_embeddings=position_embeddings_pruned,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = text_model.norm(hidden_states)

        slice_idx = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            rope_deltas=self.model.rope_deltas,
        )
