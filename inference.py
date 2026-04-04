import argparse
import io
import json
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from custom_llava_new import PrunableLlavaForConditionalGeneration
from pruner import CLIPAttentionPruner, QueryAwarePruner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="inference_results.json")
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--pruner-checkpoint", type=str, default="", help="Path to best_pruner.pt. Run without pruning if no path")
    parser.add_argument("--pruner-type", type=str, default="learned", choices=["learned", "clip-attention"],
                        help="'learned' = QueryAwarePruner, 'clip-attention' = CLIP self-attention baseline")
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Fraction of image tokens to keep.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def load_samples(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        samples = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        samples = data if isinstance(data, list) else [data]

    for idx, s in enumerate(samples):
        if "prompt" not in s or "image" not in s:
            raise ValueError(f"Sample {idx} must have 'prompt' and 'image' fields.")
    return samples


def load_image(image_ref):
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        response = requests.get(image_ref, timeout=60)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(image_ref).convert("RGB")


def load_model(args, device):
    processor = AutoProcessor.from_pretrained(args.model_id)
    model_kwargs = dict(low_cpu_mem_usage=True, torch_dtype=torch.float16)

    if args.pruner_type == "clip-attention":
        model = PrunableLlavaForConditionalGeneration.from_pretrained(
            args.model_id, **model_kwargs,
        ).to(device)
        model.eval()
        model.keep_ratio = args.keep_ratio

        pruner = CLIPAttentionPruner()
        pruner.eval()
        model.pruner = pruner
        print("CLIP attention baseline (last-layer CLS attention)")

    elif args.pruner_checkpoint:
        model = PrunableLlavaForConditionalGeneration.from_pretrained(
            args.model_id, **model_kwargs,
        ).to(device)
        model.eval()
        model.keep_ratio = args.keep_ratio

        checkpoint = torch.load(args.pruner_checkpoint, map_location=device)
        pruner_config = checkpoint["pruner_config"]
        pruner = QueryAwarePruner(**pruner_config).to(device=device, dtype=torch.float32)
        pruner.load_state_dict(checkpoint["state_dict"])
        pruner.eval()
        model.pruner = pruner
        print("Run model with learned pruner")

    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id, **model_kwargs,
        ).to(device)
        model.eval()
        print("Run baseline llava without pruner. ")

    return processor, model


def run_inference(model, processor, prompt, image, max_new_tokens, device):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    # Output only the newly generated tokens (skip the input prompt tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, latency


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples(args.input)

    processor, model = load_model(args, device)
    has_pruner = bool(args.pruner_checkpoint) or args.pruner_type == "clip-attention"
    results = []

    for idx, sample in enumerate(samples):
        print(f"[{idx+1}/{len(samples)}] {sample.get('id', idx)}")
        image = load_image(sample["image"])
        generated_text, latency = run_inference(
            model, processor, sample["prompt"], image, args.max_new_tokens, device
        )

        result = {
            "id": sample.get("id", idx),
            "prompt": sample["prompt"],
            "generated_text": generated_text,
            "latency_s": round(latency, 4),
            "pruned": has_pruner,
            "keep_ratio": args.keep_ratio if has_pruner else None,
        }

        if has_pruner and model.last_pruning_stats is not None:
            stats = model.last_pruning_stats
            original_n = stats["original_n"]
            pruned_n = stats["pruned_n"]
            reduction = (original_n - pruned_n) / original_n
            result["original_image_tokens"] = original_n
            result["pruned_image_tokens"] = pruned_n
            result["token_reduction"] = round(reduction, 4)
            print(
                f"latency={latency:.3f}s | "
                f"tokens {original_n}->{pruned_n} ({reduction*100:.1f}% reduction) | "
                f"output: {generated_text}"
            )
        else:
            print(f"latency={latency:.3f}s | output: {generated_text}")

        results.append(result)

    # Save results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} results to {output_path}")

    # Summary
    latencies = [r["latency_s"] for r in results]
    print(f"\nAvg latency: {sum(latencies)/len(latencies):.3f}s")
    if results[0].get("token_reduction") is not None:
        reductions = [r["token_reduction"] for r in results]
        print(f"Avg token reduction: {sum(reductions)/len(reductions)*100:.1f}%")

if __name__ == "__main__":
    main()
