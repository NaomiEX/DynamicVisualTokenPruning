import argparse
import io
import json
import re
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance dataset for prompt-image pairs by iteratively decoding "
            "with LLaVA and saving the full provenance tensor, image_mask, and text_mask "
            "for every decoding step."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Path to a JSON or JSONL file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--layer-range",
        type=str,
        default="0:4",
        help="Python-slice-style layer range, e.g. '0:4', '2:8', or ':'.",
    )
    parser.add_argument(
        "--decode-strategy",
        type=str,
        choices=["greedy", "sample"],
        default="greedy",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        help="Use 'eager' if you want attention maps.",
    )
    return parser.parse_args()


def parse_layer_slice(text):
    if text.strip() == ":":
        return slice(None, None)

    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid --layer-range value: {text}")

    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None

    print(f"Using attention maps from layers {start} to {end}")

    return slice(start, end)


def safe_stem(text, fallback):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return stem or fallback


def load_samples(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".jsonl":
        samples = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        if isinstance(data, list):
            samples = data
        else:
            raise ValueError("JSON input must be a list of prompt-image objects.")

    normalized = []
    for idx, sample in enumerate(samples):
        if "prompt" not in sample or "image" not in sample:
            raise ValueError(f"Sample {idx} must contain 'prompt' and 'image' fields.")
        normalized.append(
            {
                "id": str(sample.get("id", f"sample_{idx:05d}")),
                "prompt": sample["prompt"],
                "image": sample["image"],
                "metadata": sample.get("metadata", {}),
            }
        )
    return normalized


def load_image(image_ref):
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        response = requests.get(image_ref, timeout=60)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    return Image.open(image_ref).convert("RGB")


def build_model(model_id, load_in_4bit, attn_implementation):
    model_kwargs = {
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return processor, model


def prepare_base_inputs(model, processor, prompt, image):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(model.device)
    return inputs


def get_image_features(model, inputs):
    text_embeds = model.get_input_embeddings()(inputs["input_ids"])
    image_outputs = model.model.get_image_features(
        pixel_values=inputs["pixel_values"],
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        image_sizes=inputs.get("image_sizes"),
        return_dict=True,
    )

    image_features = image_outputs.pooler_output
    image_features = torch.cat(image_features, dim=0).to(text_embeds.device, text_embeds.dtype)
    if image_features.ndim == 2:
        image_features = image_features.unsqueeze(0)
    return image_features


def merge_image_features_single_example(model, input_ids, inputs_embeds, attention_mask, image_features):
    image_token_id = model.config.image_token_id
    input_ids_1 = input_ids[0]
    embeds_1 = inputs_embeds[0]
    attn_1 = attention_mask[0] if attention_mask is not None else None
    device = input_ids.device

    image_positions = torch.where(input_ids_1 == image_token_id)[0]
    if image_positions.numel() == 0:
        raise ValueError("No <image> placeholder tokens found.")

    start = image_positions[0].item()
    end = image_positions[-1].item() + 1
    expected = torch.arange(start, end, device=device)
    if not torch.equal(image_positions, expected):
        raise ValueError("This script expects the image placeholder block to be contiguous.")

    merged_embeds = torch.cat(
        [embeds_1[:start, :], image_features[0], embeds_1[end:, :]],
        dim=0,
    ).unsqueeze(0)

    merged_input_ids = torch.cat(
        [
            input_ids_1[:start],
            torch.full(
                (image_features.shape[1],),
                fill_value=image_token_id,
                dtype=input_ids_1.dtype,
                device=device,
            ),
            input_ids_1[end:],
        ],
        dim=0,
    ).unsqueeze(0)

    if attn_1 is None:
        merged_attention_mask = None
    else:
        merged_attention_mask = torch.cat(
            [
                attn_1[:start],
                torch.ones(image_features.shape[1], dtype=attn_1.dtype, device=device),
                attn_1[end:],
            ],
            dim=0,
        ).unsqueeze(0)

    merged_position_ids = torch.arange(merged_embeds.shape[1], device=device).unsqueeze(0)
    return merged_input_ids, merged_embeds, merged_attention_mask, merged_position_ids


def rollout_visual_provenance(attentions, image_mask, alpha=0.7, layer_slice=slice(0, 4)):
    selected_layers = attentions[layer_slice]
    if len(selected_layers) == 0:
        raise ValueError("No attention layers selected by --layer-range.")

    seq_len = image_mask.numel()
    num_image_tokens = int(image_mask.sum().item())
    image_positions = torch.where(image_mask)[0]

    provenance = torch.zeros(
        seq_len,
        num_image_tokens,
        device=selected_layers[0].device,
        dtype=selected_layers[0].dtype,
    )
    provenance[image_positions, torch.arange(num_image_tokens, device=provenance.device)] = 1.0

    for layer_attn in selected_layers:
        attn = layer_attn[0].mean(dim=0)
        provenance = alpha * provenance + (1.0 - alpha) * (attn @ provenance)

    return provenance


def sample_next_token(logits, decode_strategy, temperature, top_p):
    next_token_logits = logits[:, -1, :]

    if decode_strategy == "greedy":
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    probs = torch.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs * keep)
        probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


def run_decode_with_provenance(
    model,
    processor,
    prompt,
    image,
    max_new_tokens,
    alpha,
    layer_slice,
    decode_strategy,
    temperature,
    top_p,
):
    base_inputs = prepare_base_inputs(model, processor, prompt, image)
    base_input_ids = base_inputs["input_ids"]
    base_attention_mask = base_inputs.get("attention_mask")
    image_features = get_image_features(model, base_inputs)

    generated_ids = []
    steps = []
    eos_token_id = processor.tokenizer.eos_token_id

    for step_idx in range(max_new_tokens):
        if generated_ids:
            appended = torch.tensor([generated_ids], device=base_input_ids.device, dtype=base_input_ids.dtype)
            current_input_ids = torch.cat([base_input_ids, appended], dim=1)
            if base_attention_mask is None:
                current_attention_mask = None
            else:
                current_attention_mask = torch.cat(
                    [
                        base_attention_mask,
                        torch.ones_like(appended, dtype=base_attention_mask.dtype, device=base_attention_mask.device),
                    ],
                    dim=1,
                )
        else:
            current_input_ids = base_input_ids
            current_attention_mask = base_attention_mask

        current_inputs_embeds = model.get_input_embeddings()(current_input_ids)
        merged_input_ids, merged_embeds, merged_attention_mask, merged_position_ids = merge_image_features_single_example(
            model=model,
            input_ids=current_input_ids,
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            image_features=image_features,
        )

        with torch.no_grad():
            outputs = model.model.language_model(
                inputs_embeds=merged_embeds,
                attention_mask=merged_attention_mask,
                position_ids=merged_position_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
            logits = model.lm_head(outputs.last_hidden_state)

        image_mask = merged_input_ids[0] == model.config.image_token_id
        text_mask = ~image_mask
        provenance = rollout_visual_provenance(
            outputs.attentions,
            image_mask=image_mask,
            alpha=alpha,
            layer_slice=layer_slice,
        )

        next_token = sample_next_token(
            logits,
            decode_strategy=decode_strategy,
            temperature=temperature,
            top_p=top_p,
        )
        next_token_id = int(next_token.item())
        generated_ids.append(next_token_id)

        decoded_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
        steps.append(
            {
                "step_index": step_idx,
                "generated_token_id": next_token_id,
                "generated_token_text": processor.tokenizer.decode([next_token_id], skip_special_tokens=False),
                "generated_text_so_far": decoded_text,
                "sequence_input_ids": merged_input_ids[0].detach().cpu(),
                "image_mask": image_mask.detach().cpu(),
                "text_mask": text_mask.detach().cpu(),
                "provenance": provenance.detach().to(torch.float32).cpu(),
            }
        )

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return {
        "prompt": prompt,
        "generated_ids": generated_ids,
        "generated_text": processor.tokenizer.decode(generated_ids, skip_special_tokens=True),
        "steps": steps,
    }


def save_sample(output_dir, sample, result, alpha, layer_range_text):
    sample_dir = Path(output_dir) / safe_stem(sample["id"], "sample")
    sample_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": sample["id"],
        "prompt": sample["prompt"],
        "image": sample["image"],
        "metadata": sample["metadata"],
        "alpha": alpha,
        "layer_range": layer_range_text,
        "generated_ids": result["generated_ids"],
        "generated_text": result["generated_text"],
        "num_steps": len(result["steps"]),
    }

    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for step in result["steps"]:
        step_path = sample_dir / f"step_{step['step_index']:04d}.pt"
        torch.save(step, step_path)

    return sample_dir


def main():
    args = parse_args()
    # parse layers to use based on arg input
    layer_slice = parse_layer_slice(args.layer_range)

    # load input samples (TO BE CHANGED)
    # for now list of {"id": ..., "prompt":..., "image":...}
    samples = load_samples(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load in model from huggingface
    processor, model = build_model(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
    )

    manifest = []
    for sample_idx, sample in enumerate(samples):
        print(f"[{sample_idx + 1}/{len(samples)}] Processing {sample['id']}...")
        image = load_image(sample["image"])
        result = run_decode_with_provenance(
            model=model,
            processor=processor,
            prompt=sample["prompt"],
            image=image,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            layer_slice=layer_slice,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        sample_dir = save_sample(
            output_dir=output_dir,
            sample=sample,
            result=result,
            alpha=args.alpha,
            layer_range_text=args.layer_range,
        )
        manifest.append(
            {
                "id": sample["id"],
                "sample_dir": str(sample_dir),
                "num_steps": len(result["steps"]),
                "generated_text": result["generated_text"],
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved provenance dataset to {output_dir}")


if __name__ == "__main__":
    main()
