# vision-encoder and pruner forward latency for Qwen2.5-VL
import argparse
import io
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pruner import QueryAwarePruner


DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--input", default="")
    p.add_argument("--hf-dataset", default="lmms-lab/VQAv2")
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--pruner-checkpoint", default="")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPE))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="qwen_microbench_results.json")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--min-pixels", type=int, default=None,
                   help="Override processor min_pixels (Qwen2.5-VL): forces upscaling small images.")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Override processor max_pixels (Qwen2.5-VL): caps tokens for large images.")
    return p.parse_args()


def load_image(ref):
    if isinstance(ref, Image.Image):
        return ref.convert("RGB")
    if isinstance(ref, str):
        if ref.startswith(("http://", "https://")):
            r = requests.get(ref, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        return Image.open(ref).convert("RGB")
    raise TypeError(f"unsupported image ref: {type(ref)}")


def load_samples_hf(name, split, n, seed, streaming):
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=streaming).shuffle(seed=seed)
    if streaming:
        ds = ds.take(n)
    else:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for ex in ds:
        if "image" not in ex:
            raise ValueError(f"row missing 'image' key: keys={list(ex.keys())[:10]}")
        question = ex.get("question") or ex.get("prompt") or "Describe the image briefly."
        out.append({"id": str(ex.get("question_id", len(out))), "question": question, "image": ex["image"]})
    return out


def load_samples_file(path):
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    items = (
        [json.loads(l) for l in raw.splitlines() if l.strip()]
        if p.suffix.lower() == ".jsonl"
        else json.loads(raw)
    )
    if not isinstance(items, list):
        items = [items]
    out = []
    for i, s in enumerate(items):
        question = s.get("question") or s.get("prompt")
        if question is None or "image" not in s:
            raise ValueError(f"sample {i} needs 'question' (or 'prompt') and 'image'")
        out.append({"id": str(s.get("id", i)), "question": question, "image": s["image"]})
    return out


def prepare_inputs(processor, model, image, question):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)
    sync = device.type == "cuda"
    dtype = DTYPE[args.dtype]

    print(f"Loading {args.model_id} (dtype={args.dtype}, attn={args.attn_impl})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=str(device), attn_implementation=args.attn_impl,
    )
    model.eval()
    proc_kwargs = {}
    if args.min_pixels is not None:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    if proc_kwargs:
        print(f"Processor pixel bounds: {proc_kwargs}")

    text_cfg = getattr(model.config, "text_config", model.config)
    model_dim = text_cfg.hidden_size
    if args.pruner_checkpoint:
        ckpt = torch.load(args.pruner_checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("pruner_config", {"dim": model_dim})
        if cfg.get("dim") != model_dim:
            raise ValueError(f"pruner ckpt dim {cfg.get('dim')} != model {model_dim}")
        pruner = QueryAwarePruner(**cfg).to(device=device, dtype=dtype)
        pruner.load_state_dict(ckpt["state_dict"])
        print(f"Loaded learned pruner from {args.pruner_checkpoint}")
    else:
        pruner = QueryAwarePruner(dim=model_dim).to(device=device, dtype=dtype)
        print(f"Random-init pruner at dim={model_dim} (timing only).")
    pruner.eval()

    samples = load_samples_file(args.input) if args.input else load_samples_hf(
        args.hf_dataset, args.hf_split, args.num_samples, args.seed, args.streaming
    )
    print(f"Loaded {len(samples)} samples | keep_ratio={args.keep_ratio}")

    image_token_id = model.config.image_token_id

    if args.warmup > 0 and samples:
        print(f"Warmup ({args.warmup} iters)...")
        warm = prepare_inputs(processor, model, load_image(samples[0]["image"]), samples[0]["question"])
        with torch.no_grad():
            for _ in range(args.warmup):
                feats = model.model.get_image_features(warm["pixel_values"], warm["image_grid_thw"])
                v = torch.cat(feats, dim=0).unsqueeze(0).to(dtype)
                t_mask = warm["input_ids"][0] != image_token_id
                t_embeds = model.get_input_embeddings()(warm["input_ids"][0][t_mask]).unsqueeze(0).to(dtype)
                pruner(v, t_embeds, keep_ratio=args.keep_ratio)
        if sync:
            torch.cuda.synchronize()

    results: List[Dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        try:
            image = load_image(s["image"])
            inputs = prepare_inputs(processor, model, image, s["question"])
            num_image_tokens = int((inputs["input_ids"][0] == image_token_id).sum().item())

            with torch.no_grad():
                # Vision encoder
                if sync:
                    torch.cuda.synchronize()
                t = time.perf_counter()
                feats = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
                if sync:
                    torch.cuda.synchronize()
                vision_ms = (time.perf_counter() - t) * 1000.0

                v = torch.cat(feats, dim=0).unsqueeze(0).to(dtype)
                t_mask = inputs["input_ids"][0] != image_token_id
                t_embeds = model.get_input_embeddings()(inputs["input_ids"][0][t_mask]).unsqueeze(0).to(dtype)

                # Pruner
                if sync:
                    torch.cuda.synchronize()
                t = time.perf_counter()
                _, keep_idx = pruner(v, t_embeds, keep_ratio=args.keep_ratio)
                if sync:
                    torch.cuda.synchronize()
                pruner_ms = (time.perf_counter() - t) * 1000.0

            kept = int(keep_idx.shape[1])
            row = {
                "id": s.get("id"),
                "num_image_tokens": num_image_tokens,
                "kept_image_tokens": kept,
                "vision_encoder_ms": round(vision_ms, 3),
                "pruner_ms": round(pruner_ms, 3),
            }
            results.append(row)
            print(f"[{i}/{len(samples)}] id={row['id']} | img_tok={num_image_tokens}->{kept} | "
                  f"vision={vision_ms:.2f}ms pruner={pruner_ms:.2f}ms")
        except Exception as e:
            print(f"  [{i}] error: {e}")
            results.append({"id": s.get("id"), "error": str(e)})

    ok = [r for r in results if "vision_encoder_ms" in r]
    summary = {
        "num_samples": len(samples),
        "num_ok": len(ok),
        "avg_vision_encoder_ms": round(mean(r["vision_encoder_ms"] for r in ok), 3) if ok else None,
        "avg_pruner_ms": round(mean(r["pruner_ms"] for r in ok), 3) if ok else None,
        "avg_num_image_tokens": round(mean(r["num_image_tokens"] for r in ok), 2) if ok else None,
        "avg_kept_image_tokens": round(mean(r["kept_image_tokens"] for r in ok), 2) if ok else None,
    }
    payload = {
        "config": {
            "model_id": args.model_id,
            "keep_ratio": args.keep_ratio,
            "pruner_checkpoint": args.pruner_checkpoint or None,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "device": str(device),
            "warmup": args.warmup,
            "source": args.input or f"{args.hf_dataset}:{args.hf_split}",
            "num_samples_requested": args.num_samples,
        },
        "summary": summary,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
