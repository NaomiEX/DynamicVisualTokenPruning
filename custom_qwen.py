"""Prunable Qwen2.5-VL subclass.

Mirrors custom_llava_new.PrunableLlavaForConditionalGeneration: attach a
pruner instance and a keep_ratio, then call generate() normally. The model
runs the visual encoder, the pruner scores tokens, and kept image features
are spliced into a pruned (input_ids, attention_mask, inputs_embeds) trio
*before* the LM sees the sequence. The model's own get_rope_index then
computes M-RoPE for the pruned sequence, so we never construct
position_ids or rope_deltas by hand.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration


class PrunableQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.pruner = None
        self.keep_ratio: float = 0.5
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
            self.pruner is not None
            and is_prefill
            and pixel_values is not None
            and input_ids is not None
            and image_grid_thw is not None
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

        image_token_id = self.config.image_token_id

        image_features_list = self.model.get_image_features(pixel_values, image_grid_thw)
        if hasattr(image_features_list, "pooler_output"):
            image_features_list = image_features_list.pooler_output
        full_image_embeds = torch.cat(image_features_list, dim=0)  # [N_img, D]

        full_text_embeds = self.get_input_embeddings()(input_ids)  # [B, S, D]
        non_image_mask = input_ids[0] != image_token_id
        text_only_embeds = full_text_embeds[0][non_image_mask].unsqueeze(0)  # [1, T, D]

        try:
            pruner_dtype = next(self.pruner.parameters()).dtype
        except StopIteration:
            pruner_dtype = full_image_embeds.dtype
        with torch.no_grad():
            _, keep_idx = self.pruner(
                full_image_embeds.unsqueeze(0).to(pruner_dtype),
                text_only_embeds.to(pruner_dtype),
                keep_ratio=self.keep_ratio,
            )
        keep_idx = keep_idx[0]
        K = int(keep_idx.numel())
        N_img = int(full_image_embeds.shape[0])

        self.last_pruning_stats = {"original_n": N_img, "pruned_n": K}

        image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        keep_seq = torch.ones_like(input_ids[0], dtype=torch.bool)
        keep_seq[image_positions] = False
        keep_seq[image_positions[keep_idx]] = True
        pruned_input_ids = input_ids[:, keep_seq]
        pruned_attn_mask = attention_mask[:, keep_seq] if attention_mask is not None else None

        new_embeds = full_text_embeds[:, keep_seq].clone()
        kept_image_in_new = pruned_input_ids[0] == image_token_id
        new_embeds[0, kept_image_in_new] = full_image_embeds[keep_idx].to(new_embeds.dtype)

        # Synthesize a grid_thw whose post-merger token count equals K, so
        # get_rope_index assigns M-RoPE positions consistent with the pruned
        # sequence. Pre-merger H*W must equal K * spatial_merge_size**2.
        sm = self.config.vision_config.spatial_merge_size
        fake_grid_thw = torch.tensor(
            [[1, sm, sm * K]],
            device=image_grid_thw.device,
            dtype=image_grid_thw.dtype,
        )

        # Force recompute: previous self.model.rope_deltas (if any) was for
        # the unpruned sequence and would corrupt decode.
        self.model.rope_deltas = None

        return super().forward(
            input_ids=pruned_input_ids,
            attention_mask=pruned_attn_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=new_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=None,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=fake_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=None,
            cache_position=None,
            second_per_grid_ts=second_per_grid_ts,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, *args, **kwargs
        )
        pkv = model_kwargs.get("past_key_values")
        am = model_kwargs.get("attention_mask")
        if pkv is not None and am is not None and hasattr(pkv, "get_seq_length"):
            kv_len = pkv.get_seq_length()
            if kv_len > 0 and am.shape[1] != kv_len:
                model_kwargs["attention_mask"] = am.new_ones((am.shape[0], kv_len))
        return model_kwargs
