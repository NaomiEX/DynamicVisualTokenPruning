import argparse
import importlib
import io
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from custom_llava_new import PrunableLlavaForConditionalGeneration
from pruner import QueryAwarePruner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 timing eval for baseline or multiple pruner variants on local JSON or HF datasets"
    )
    parser.add_argument("--input", type=str, default="", help="Path to JSON/JSONL samples (optional)")
    parser.add_argument("--hf-dataset", type=str, default="", help="HF dataset name, e.g. lmms-lab/VQAv2")
    parser.add_argument("--hf-split", type=str, default="validation")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="stage1_timing_results.json")
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")

    parser.add_argument(
        "--pruner-type",
        type=str,
        default="none",
        choices=["none", "learned", "llm-attention", "clip-attention"],
        help="Pruner to evaluate: none=baseline, learned=checkpointed QueryAwarePruner, "
             "llm-attention=LLM attention baseline, clip-attention=CLIP attention baseline",
    )
    parser.add_argument(
        "--pruner-checkpoint",
        type=str,
        default="",
        help="Path to learned pruner checkpoint (.pt). Required when --pruner-type learned",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument(
        "--llm-layers",
        type=int,
        default=1,
        help="Number of layers to use for LLM-attention baseline pruner",
    )

    parser.add_argument("--streaming", action="store_true", help="Stream HF dataset instead of downloading fully")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_samples_from_file(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        samples = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        samples = data if isinstance(data, list) else [data]

    for idx, s in enumerate(samples):
        if "prompt" not in s or "image" not in s:
            raise ValueError(f"Sample {idx} must contain 'prompt' and 'image'.")
    return samples


def load_samples_from_hf(dataset_name: str, split: str, num_samples: int, seed: int, streaming: bool = False) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    ds = ds.shuffle(seed=seed)

    if streaming:
        ds = ds.take(num_samples)
    else:
        take_n = min(num_samples, len(ds))
        ds = ds.select(range(take_n))

    samples: List[Dict[str, Any]] = []
    for ex in ds:
        if "question" not in ex or "image" not in ex:
            raise ValueError(
                f"Dataset row missing expected keys. Found keys: {list(ex.keys())[:20]}"
            )
        sample = {
            "id": str(ex.get("question_id", len(samples))),
            "prompt": f"USER: <image>\n{ex['question']}\nASSISTANT:",
            "image": ex["image"],
        }
        if "multiple_choice_answer" in ex:
            sample["answer"] = ex["multiple_choice_answer"]
        elif "answer" in ex:
            sample["answer"] = ex["answer"]
        samples.append(sample)
    return samples


def load_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input:
        return load_samples_from_file(args.input)
    if args.hf_dataset:
        return load_samples_from_hf(args.hf_dataset, args.hf_split, args.num_samples, args.seed, streaming=args.streaming)
    raise ValueError("Provide either --input or --hf-dataset")


def load_image(image_ref: Any) -> Image.Image:
    if isinstance(image_ref, Image.Image):
        return image_ref.convert("RGB")
    if isinstance(image_ref, str):
        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(image_ref, timeout=60, headers=headers)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return Image.open(image_ref).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_ref)}")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def import_pruner_class(class_name: str):
    candidates = [
        ("pruner", class_name),
        ("llm_attention_pruner", class_name),
        ("clip_attention_pruner", class_name),
        ("attention_pruner", class_name),
        ("baseline_pruners", class_name),
    ]
    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except ImportError:
            continue
    raise ImportError(
        f"Could not import {class_name}. Looked in: "
        + ", ".join([m for m, _ in candidates])
    )


def load_model(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(args.model_id)
    model_dtype = get_torch_dtype(args.dtype)
    model_kwargs = dict(low_cpu_mem_usage=True, torch_dtype=model_dtype)

    if args.pruner_type == "none":
        model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).to(device)
        model.eval()
        mode = "baseline"
        return processor, model, mode

    model = PrunableLlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).to(device)
    model.eval()
    model.keep_ratio = args.keep_ratio

    if args.pruner_type == "learned":
        if not args.pruner_checkpoint:
            raise ValueError("--pruner-checkpoint is required when --pruner-type learned")

        checkpoint = torch.load(args.pruner_checkpoint, map_location="cpu", weights_only=False)
        pruner_cfg = checkpoint["pruner_config"]
        pruner = QueryAwarePruner(**pruner_cfg).to(device=device, dtype=model_dtype)
        pruner.load_state_dict(checkpoint["state_dict"])
        pruner.eval()
        model.pruner = pruner
        mode = "learned-pruner"

    elif args.pruner_type == "llm-attention":
        LLMAttentionPruner = import_pruner_class("LLMAttentionPruner")
        pruner = LLMAttentionPruner(num_layers=args.llm_layers)
        pruner.eval()
        model.pruner = pruner
        mode = f"llm-attention-{args.llm_layers}layer"

    elif args.pruner_type == "clip-attention":
        CLIPAttentionPruner = import_pruner_class("CLIPAttentionPruner")
        pruner = CLIPAttentionPruner()
        pruner.eval()
        model.pruner = pruner
        mode = "clip-attention"

    else:
        raise ValueError(f"Unknown pruner type: {args.pruner_type}")

    return processor, model, mode


def greedy_or_sample(logits: torch.Tensor, do_sample: bool, temperature: float, top_p: float) -> torch.Tensor:
    next_token_logits = logits[:, -1, :]
    if not do_sample:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature must be > 0 when sampling")
    next_token_logits = next_token_logits / temperature
    probs = torch.softmax(next_token_logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_idx, -1, sampled)

    return torch.multinomial(probs, num_samples=1)


def extract_pruning_stats(outputs: Any, model: Any) -> Optional[Dict[str, Any]]:
    old_n = getattr(outputs, "original_image_token_count", None)
    new_n = getattr(outputs, "pruned_image_token_count", None)
    keep_idx = getattr(outputs, "pruned_keep_indices", None)

    if old_n is None or new_n is None:
        stats = getattr(model, "last_pruning_stats", None)
        if stats is not None:
            old_n = stats.get("original_n", stats.get("original_image_tokens"))
            new_n = stats.get("pruned_n", stats.get("pruned_image_tokens"))
            keep_idx = stats.get("keep_indices", stats.get("pruned_keep_indices"))

    if old_n is None or new_n is None:
        return None
    if isinstance(old_n, torch.Tensor):
        old_n = int(old_n.item())
    if isinstance(new_n, torch.Tensor):
        new_n = int(new_n.item())
    if isinstance(keep_idx, torch.Tensor):
        keep_idx = keep_idx.detach().cpu().tolist()

    return {
        "original_image_tokens": int(old_n),
        "pruned_image_tokens": int(new_n),
        "token_reduction": float((int(old_n) - int(new_n)) / int(old_n)) if int(old_n) else None,
        "keep_indices": keep_idx,
    }


@torch.no_grad()
def run_single_sample(model, processor, sample: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    image = load_image(sample["image"])
    prompt = sample["prompt"]
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

    eos_token_id = processor.tokenizer.eos_token_id
    generated_token_ids: List[int] = []
    per_token_latencies_ms: List[float] = []

    sync_if_cuda(device)
    t0 = time.perf_counter()
    outputs = model(**inputs, use_cache=True, return_dict=True)
    next_token = greedy_or_sample(outputs.logits, args.do_sample, args.temperature, args.top_p)
    sync_if_cuda(device)
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    per_token_latencies_ms.append(ttft_ms)
    generated_token_ids.append(int(next_token.item()))
    pruning_stats = extract_pruning_stats(outputs, model)

    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs["input_ids"], device=device)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)], dim=1
    )

    past_key_values = outputs.past_key_values
    cur_input_ids = next_token

    for _ in range(1, args.max_new_tokens):
        if eos_token_id is not None and generated_token_ids[-1] == eos_token_id:
            break
        sync_if_cuda(device)
        step_t0 = time.perf_counter()
        step_outputs = model(
            input_ids=cur_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            pixel_values=None,
            use_cache=True,
            return_dict=True,
        )
        next_token = greedy_or_sample(step_outputs.logits, args.do_sample, args.temperature, args.top_p)
        sync_if_cuda(device)
        step_ms = (time.perf_counter() - step_t0) * 1000.0
        per_token_latencies_ms.append(step_ms)

        generated_token_ids.append(int(next_token.item()))
        cur_input_ids = next_token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        past_key_values = step_outputs.past_key_values

    total_e2e_ms = sum(per_token_latencies_ms)
    inter_token_latencies = per_token_latencies_ms[1:]
    avg_itl_ms = mean(inter_token_latencies) if inter_token_latencies else None
    generated_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    image_ref = sample["image"]
    image_meta = image_ref if isinstance(image_ref, str) else f"<{type(image_ref).__name__}>"
    is_pruned_mode = args.pruner_type != "none"

    result: Dict[str, Any] = {
        "id": sample.get("id"),
        "prompt": prompt,
        "image": image_meta,
        "generated_text": generated_text,
        "generated_token_count": len(generated_token_ids),
        "e2e_latency_ms": round(total_e2e_ms, 3),
        "ttft_ms": round(ttft_ms, 3),
        "avg_inter_token_latency_ms": round(avg_itl_ms, 3) if avg_itl_ms is not None else None,
        "per_token_latencies_ms": [round(x, 3) for x in per_token_latencies_ms],
        "pruned": is_pruned_mode,
        "pruner_type": args.pruner_type,
        "keep_ratio": args.keep_ratio if is_pruned_mode else None,
    }

    if args.pruner_type == "learned":
        result["pruner_checkpoint"] = args.pruner_checkpoint
    if args.pruner_type == "llm-attention":
        result["llm_layers"] = args.llm_layers

    if "answer" in sample:
        gold = str(sample["answer"]).strip().lower()
        pred = generated_text.strip().lower()
        result["answer"] = sample["answer"]
        result["exact_match"] = int(pred == gold)

    if pruning_stats is not None:
        result.update(pruning_stats)

    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": len(results),
        "avg_e2e_latency_ms": round(mean(r["e2e_latency_ms"] for r in results), 3),
        "avg_ttft_ms": round(mean(r["ttft_ms"] for r in results), 3),
    }
    itl_vals = [r["avg_inter_token_latency_ms"] for r in results if r["avg_inter_token_latency_ms"] is not None]
    if itl_vals:
        summary["avg_inter_token_latency_ms"] = round(mean(itl_vals), 3)
    reductions = [r.get("token_reduction") for r in results if r.get("token_reduction") is not None]
    if reductions:
        summary["avg_token_reduction"] = round(mean(reductions), 4)
    ems = [r["exact_match"] for r in results if "exact_match" in r]
    if ems:
        summary["exact_match"] = round(sum(ems) / len(ems), 4)
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    samples = load_samples(args)
    processor, model, mode = load_model(args, device)

    all_results: List[Dict[str, Any]] = []
    source_desc = args.input if args.input else f"{args.hf_dataset}:{args.hf_split}"
    print(f"Loaded {len(samples)} samples from {source_desc} | mode={mode} | device={device}")

    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] id={sample.get('id', idx)}")
        result = run_single_sample(model, processor, sample, args, device)
        all_results.append(result)

        msg = (
            f"  e2e={result['e2e_latency_ms']:.1f}ms | "
            f"ttft={result['ttft_ms']:.1f}ms | "
            f"itl={result['avg_inter_token_latency_ms'] if result['avg_inter_token_latency_ms'] is not None else 'NA'}ms | "
            f"tokens={result['generated_token_count']} | output={result['generated_text']!r}"
        )
        if result.get("token_reduction") is not None:
            msg += f" | reduction={result['token_reduction']*100:.1f}%"
        print(msg)

    payload = {
        "config": {
            "source": source_desc,
            "model_id": args.model_id,
            "mode": mode,
            "pruner_type": args.pruner_type,
            "pruner_checkpoint": args.pruner_checkpoint if args.pruner_type == "learned" else None,
            "llm_layers": args.llm_layers if args.pruner_type == "llm-attention" else None,
            "keep_ratio": args.keep_ratio if args.pruner_type != "none" else None,
            "max_new_tokens": args.max_new_tokens,
            "device": str(device),
            "dtype": args.dtype,
        },
        "summary": summarize(all_results),
        "results": all_results,
    }

    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {out}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
