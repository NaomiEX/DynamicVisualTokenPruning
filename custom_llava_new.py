import torch
from transformers import (
    LlavaForConditionalGeneration,
)

from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast


class PrunableLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    # starter prototype
    # IMPORTANT ASSUMPTIONS:
    # batch size 1, single image

    def __init__(self, config):
        super().__init__(config)
        # Initialize the pruner separately to bypass the base model's 4-bit quantization,
        # and keep the pruner in float16 to ensure high-precision gradients.
        self.pruner = None 

    def prune_image_features(
        self,
        image_features,
        inputs_embeds,
        keep_ratio=0.5
    ):
        # assumptions: image features are [1, N, D]
        # returns:
        #   - pruned_features: [1, K, D]
        #   - keep_idx: [K]
        # B, N, D = image_features.shape
        # assert B == 1, "This starter patch only supports batch size 1"

        # # calculates n features to keep
        # keep_n = max(1, int(N*keep_ratio)) 

        # # keep first n tokens
        # keep_idx = torch.arange(keep_n, device=image_features.device)

        # pruned = image_features[:, keep_idx, :] # [1, K, D]

        # Use cross attention and MLP to compute importance scores
        pruned, keep_idx = self.pruner(image_features, inputs_embeds, keep_ratio=keep_ratio)
        return pruned, keep_idx

    def _merge_pruned_image_features_single_example(
        self,
        input_ids, # token ids of the text prompt, including <image> placeholder tokens [B, L]
        inputs_embeds, # embeddings (including text + image) # [B, L, D]
        attention_mask, # tells the model which positions are real tokens
        image_features, # the pruned image features
    ):
        assert image_features.shape[0] == 1, "expecting batch size 1"

        device=input_ids.device

        input_ids_1 = input_ids[0] # [L]
        embeds_1 = inputs_embeds[0] # [L, D]
        attn_1 = attention_mask[0] if attention_mask is not None else None

        image_token_id = self.config.image_token_id
        # this gives us a mask to be able to retrieve the image tokens specifically
        image_mask = (input_ids_1 == image_token_id) 
        image_positions = torch.where(image_mask)[0]

        if image_positions.numel() == 0:
            raise ValueError("No image placeholder tokens found in input_ids.")
        
        # We assume the image placeholders are one contiguous block.
        # get start and end idx
        start = image_positions[0].item()
        end = image_positions[-1].item() + 1
        original_n = image_positions.numel()
        pruned_n = image_features.shape[1]

        # Sanity check: checks that the image tokens are contiguous
        expected = torch.arange(start, end, device=device)
        if not torch.equal(image_positions, expected):
            raise ValueError(
                "Starter patch expects image placeholder tokens to be contiguous."
            )
        
        # Split text into prefix / image-block / suffix
        # we dont touch the prefix or suffix, just modify the image block part
        prefix_embeds = embeds_1[:start, :]                # [A, D]
        suffix_embeds = embeds_1[end:, :]                  # [B, D]

        pruned_img_embeds = image_features[0]              # [K, D]

        new_embeds = torch.cat(
            [prefix_embeds, pruned_img_embeds, suffix_embeds], dim=0
        )  # [A + K + B, D]


        # Build position ids from scratch
        new_seq_len = new_embeds.shape[0]
        new_position_ids = torch.arange(
            new_seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        # Also rebuild input_ids for consistency
        # same idea, split attention mask into prefix / image block / suffix
        # for the image block part just need a block of size K filled with image_token_id constant value
        prefix_ids = input_ids_1[:start]
        suffix_ids = input_ids_1[end:]
        pruned_image_ids = torch.full(
            (pruned_n,),
            fill_value=image_token_id,
            dtype=input_ids_1.dtype,
            device=device
        )
        new_input_ids = torch.cat(
            [prefix_ids, pruned_image_ids, suffix_ids], dim=0
        ).unsqueeze(0)

        # Build new attention mask
        if attn_1 is not None:
            # same idea, split attention mask into prefix / image block / suffix
            prefix_attn = attn_1[:start]
            suffix_attn = attn_1[end:]
            pruned_attn = torch.ones(pruned_n, dtype=attn_1.dtype, device=device)
            new_attention_mask = torch.cat(
                [prefix_attn, pruned_attn, suffix_attn], dim=0
            ).unsqueeze(0)
        else:
            new_attention_mask = None

        return new_input_ids, new_embeds.unsqueeze(0), new_attention_mask, new_position_ids, original_n, pruned_n


    def forward(
        self,
        # params from LLavaForConditionalGeneration
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        image_sizes: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,

        # new params
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        # params from LLavaForConditionalGeneration (into kwargs)
        # cache_position: torch.LongTensor | None = None,

        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## for decode steps, the same image tokens are reused so no pruning required, run normal behaviour 
        if pixel_values is None:
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                image_sizes=image_sizes,
                return_dict=return_dict,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        
        ## first iteration (prefill with new image tokens)

        if (input_ids is None) == (inputs_embeds is None):
            if inputs_embeds is None:
                raise ValueError("Provide input_ids for the starter patch.")
            else:
                raise ValueError("Starter patch expects input_ids, not only inputs_embeds.")

        # 1) Text embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2) Get image features (Note this entire section is the same as this part of LlavaModel's forward)
        image_outputs = self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
            return_dict=True
        )

        image_features = image_outputs.pooler_output
        image_features = torch.cat(image_features, dim=0)\
                              .to(inputs_embeds.device, inputs_embeds.dtype)

        if image_features.ndim == 2:
            image_features = image_features.unsqueeze(0)
        
        # 3) ========= PRUNE IMAGE FEATURES HERE ===========

        pruned_image_features, keep_idx = self.prune_image_features(
            image_features, inputs_embeds, keep_ratio=0.5
        )

        # 4) rebuild sequence
        new_input_ids, new_inputs_embeds, new_attention_mask, new_position_ids, old_n, new_n = \
            self._merge_pruned_image_features_single_example(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                image_features=pruned_image_features,
            )
        
        # 5) Call the language model directly with the rebuilt embeddings
        outputs = self.model.language_model(
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        ## this part is similar to LlavaForConditionalGeneration after it calls the model

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Starter patch does not remap labels after pruning.
            # Fine for inference; for training you'd need to rebuild labels too.
            raise NotImplementedError("Label remapping is not implemented in this starter patch. (Training not supported!)")

        if not return_dict:
            return (logits,) + outputs[1:]

        result = LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=pruned_image_features,
        )

        # Extra debug info
        result.original_image_token_count = old_n
        result.pruned_image_token_count = new_n
        result.pruned_keep_indices = keep_idx
        result.rebuilt_input_ids = new_input_ids

        return result