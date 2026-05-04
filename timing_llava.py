# Timing eval for LLaVA-1.5 with pruning options
# TTFT, TBT, Total
# Pruner and vision step timings are in microbenchmark scripts
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
from transformers import AutoProcessor, LlavaForConditionalGeneration, LogitsProcessor, LogitsProcessorList

from custom_llava_new import PrunableLlavaForConditionalGeneration
from pruner import CLIPAttentionPruner, LLMAttentionPruner, QueryAwarePruner


DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--input", default="")
    p.add_argument("--hf-dataset", default="lmms-lab/VQAv2")
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--pruner-type", choices=["none", "random", "learned", "clip-attention", "llm-attention"], default="none")
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--pruner-checkpoint", default="")
    p.add_argument("--llm-layers", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="float16", choices=list(DTYPE))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="llava_timing_results.json")
    p.add_argument("--warmup", type=int, default=2)
    return p.parse_args()


class TimingLogitsProcessor(LogitsProcessor):
    def __init__(self, sync_cuda: bool):
        self.times: List[float] = []
        self.sync_cuda = sync_cuda

    def __call__(self, input_ids, scores):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return scores


class RandomPruner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dtype_anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x_v, x_t, keep_ratio=0.5):
        B, N, _ = x_v.shape
        k = max(1, int(N * keep_ratio))
        idx = torch.stack([torch.randperm(N, device=x_v.device)[:k] for _ in range(B)], dim=0)
        idx, _ = torch.sort(idx, dim=1)
        return x_v.gather(1, idx.unsqueeze(-1).expand(-1, -1, x_v.shape[-1])), idx


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
        s = {"id": str(ex.get("question_id", len(out))), "question": question, "image": ex["image"]}
        if "multiple_choice_answer" in ex:
            s["answer"] = ex["multiple_choice_answer"]
        elif "answer" in ex:
            s["answer"] = ex["answer"]
        out.append(s)
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
        out.append({"id": str(s.get("id", i)), "question": question, "image": s["image"], "answer": s.get("answer")})
    return out


def prepare_inputs(processor, model, image, question):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}


@torch.no_grad()
def time_generate(model, generate_kwargs, max_new_tokens, sync_cuda):
    timing = TimingLogitsProcessor(sync_cuda)
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    output_ids = model.generate(
        **generate_kwargs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        logits_processor=LogitsProcessorList([timing]),
    )
    if sync_cuda:
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return output_ids, timing.times, t0, t_end


def summarize_timing(t0, times, t_end):
    if not times:
        return None
    ttft_ms = (times[0] - t0) * 1000.0
    inter = [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)]
    return {
        "ttft_ms": round(ttft_ms, 3),
        "avg_tbt_ms": round(mean(inter), 3) if inter else None,
        "total_latency_ms": round((t_end - t0) * 1000.0, 3),
        "num_decode_steps": len(times),
        "per_token_latencies_ms": [round(x, 3) for x in inter],
    }


def build_model(args, device, dtype):
    if args.pruner_type == "none":
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        return model

    model = PrunableLlavaForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.keep_ratio = args.keep_ratio

    if args.pruner_type == "clip-attention":
        pruner = CLIPAttentionPruner()
    elif args.pruner_type == "llm-attention":
        pruner = LLMAttentionPruner(num_layers=args.llm_layers)
    elif args.pruner_type == "random":
        pruner = RandomPruner().to(device=device, dtype=dtype)
    else:  # learned
        text_cfg = getattr(model.config, "text_config", model.config)
        model_dim = text_cfg.hidden_size
        if args.pruner_checkpoint:
            ckpt = torch.load(args.pruner_checkpoint, map_location="cpu", weights_only=False)
            cfg = ckpt.get("pruner_config", {"dim": model_dim})
            if cfg.get("dim") != model_dim:
                raise ValueError(
                    f"pruner checkpoint dim {cfg.get('dim')} != model hidden size {model_dim}"
                )
            pruner = QueryAwarePruner(**cfg).to(device=device, dtype=dtype)
            pruner.load_state_dict(ckpt["state_dict"])
            print(f"Loaded learned pruner from {args.pruner_checkpoint}")
        else:
            pruner = QueryAwarePruner(dim=model_dim).to(device=device, dtype=dtype)
            print(
                "WARN: --pruner-type learned with no checkpoint; using random init "
                f"at dim={model_dim}. Timings are realistic but generated text is not."
            )

    pruner.eval()
    model.pruner = pruner
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)
    sync = device.type == "cuda"
    dtype = DTYPE[args.dtype]

    print(f"Loading {args.model_id} (dtype={args.dtype})")
    model = build_model(args, device, dtype)
    processor = AutoProcessor.from_pretrained(args.model_id)

    samples = load_samples_file(args.input) if args.input else load_samples_hf(
        args.hf_dataset, args.hf_split, args.num_samples, args.seed, args.streaming
    )
    print(f"Loaded {len(samples)} samples | mode={args.pruner_type} | keep_ratio={args.keep_ratio}")

    if args.warmup > 0 and samples:
        print(f"Warmup ({args.warmup} iters)...")
        warm = prepare_inputs(processor, model, load_image(samples[0]["image"]), samples[0]["question"])
        with torch.no_grad():
            for _ in range(args.warmup):
                model.generate(**warm, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        if sync:
            torch.cuda.synchronize()

    results: List[Dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        try:
            image = load_image(s["image"])
            inputs = prepare_inputs(processor, model, image, s["question"])
            input_len = int(inputs["input_ids"].shape[1])

            out_ids, times, t0, t_end = time_generate(model, inputs, args.max_new_tokens, sync)
            timing = summarize_timing(t0, times, t_end) or {}

            stats = getattr(model, "last_pruning_stats", None)
            if args.pruner_type != "none" and stats is not None:
                original = stats["original_n"]
                kept = stats["pruned_n"]
            else:
                original = None
                kept = None

            gen_only = out_ids[0][input_len:]
            text = processor.tokenizer.decode(gen_only, skip_special_tokens=True)

            row: Dict[str, Any] = {
                "id": s.get("id"),
                "question": s["question"],
                "input_len": input_len,
                "num_image_tokens": original,
                "kept_image_tokens": kept,
                "tokens_pruned": (original - kept) if (original is not None and kept is not None) else None,
                "pruner_type": args.pruner_type,
                "keep_ratio": args.keep_ratio if args.pruner_type != "none" else None,
                "ttft_ms": timing.get("ttft_ms"),
                "avg_tbt_ms": timing.get("avg_tbt_ms"),
                "total_latency_ms": timing.get("total_latency_ms"),
                "num_decode_steps": timing.get("num_decode_steps"),
                "per_token_latencies_ms": timing.get("per_token_latencies_ms"),
                "generated_text": text,
                "generated_token_count": int(gen_only.numel()),
            }
            if "answer" in s and s["answer"] is not None:
                row["answer"] = s["answer"]
                row["exact_match"] = int(text.strip().lower() == str(s["answer"]).strip().lower())

            results.append(row)
            print(
                f"[{i}/{len(samples)}] id={row['id']} | "
                f"img_tok={original}->{kept} | "
                f"ttft={row['ttft_ms']}ms tbt={row['avg_tbt_ms']}ms total={row['total_latency_ms']}ms"
            )
        except Exception as e:
            print(f"  [{i}] error: {e}")
            results.append({"id": s.get("id"), "error": str(e)})

    ok = [r for r in results if "ttft_ms" in r and r["ttft_ms"] is not None]
    summary = {
        "num_samples": len(samples),
        "num_ok": len(ok),
        "avg_ttft_ms": round(mean(r["ttft_ms"] for r in ok), 3) if ok else None,
        "avg_tbt_ms": round(
            mean(r["avg_tbt_ms"] for r in ok if r.get("avg_tbt_ms") is not None), 3
        ) if ok else None,
        "avg_total_latency_ms": round(mean(r["total_latency_ms"] for r in ok), 3) if ok else None,
    }
    if ok and ok[0].get("num_image_tokens") is not None:
        summary["avg_num_image_tokens"] = round(mean(r["num_image_tokens"] for r in ok), 2)
        summary["avg_kept_image_tokens"] = round(mean(r["kept_image_tokens"] for r in ok), 2)
        summary["avg_tokens_pruned"] = round(mean(r["tokens_pruned"] for r in ok), 2)
    ems = [r["exact_match"] for r in ok if "exact_match" in r]
    if ems:
        summary["exact_match"] = round(sum(ems) / len(ems), 4)

    payload = {
        "config": {
            "model_id": args.model_id,
            "pruner_type": args.pruner_type,
            "keep_ratio": args.keep_ratio if args.pruner_type != "none" else None,
            "pruner_checkpoint": args.pruner_checkpoint or None,
            "llm_layers": args.llm_layers if args.pruner_type == "llm-attention" else None,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
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
