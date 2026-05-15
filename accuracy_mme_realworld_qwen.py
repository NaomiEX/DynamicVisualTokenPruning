# Accuracy + timing eval for Qwen2.5-VL on MME-RealWorld-Lite (camera-perception subset)
# Dataset: yifanzhang114/MME-RealWorld-lite-lmms-eval
# Filtered to category in {"Perception/Autonomous_Driving", "Perception/Monitoring"} -> 669 rows
import argparse
import base64
import binascii
import io
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList

from custom_qwen import PrunableQwen2_5_VLForConditionalGeneration
from fastv_qwen import FastVQwen2_5_VLForConditionalGeneration
from pruner import QueryAwarePruner


DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

CAMERA_CATEGORIES = {"Perception/Autonomous_Driving", "Perception/Monitoring"}

_CHOICE_RE = re.compile(r"^\(?\s*([ABCDE])\s*\)?(?:\s|\.|,|:|$)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--hf-dataset", default="yifanzhang114/MME-RealWorld-lite-lmms-eval")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--categories", default=",".join(sorted(CAMERA_CATEGORIES)),
                   help="Comma-separated 'category' values to keep. Default = camera-perception subset.")
    p.add_argument("--num-samples", type=int, default=0,
                   help="Cap samples after filtering (0 = use all matched rows).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle filtered rows before truncating with --num-samples.")
    p.add_argument("--pruner-type", choices=["none", "random", "learned", "fastv"], default="none")
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--pruner-checkpoint", default="")
    p.add_argument("--fastv-k", type=int, default=2,
                   help="FastV pruning layer index (only used when --pruner-type fastv).")
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="Answers are a single letter, so 8 is plenty.")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPE))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="mme_realworld_lite_results.json")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28,
                   help="Processor min_pixels (Qwen2.5-VL). Default matches prior MME pipelines.")
    p.add_argument("--max-pixels", type=int, default=1024 * 28 * 28,
                   help="Processor max_pixels (Qwen2.5-VL). Default matches prior MME pipelines.")
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


class StageTimer:
    """Times the vision tower and pure LLM-decoder prefill via forward hooks.

    `vision_ms`: ViT forward only (model.model.visual).
    `prefill_ms`: time from entry of layers[0] to exit of text_model.norm on the
    first invocation — i.e. the LLM decoder prefill, excluding vision encoding,
    embed/rope/mask construction, and the lm_head projection. Hooks only record
    the first call after each reset(), so decode steps don't overwrite.
    """

    def __init__(self, sync_cuda: bool):
        self.sync = sync_cuda
        self._hooks: List[Any] = []
        self.reset()

    def reset(self):
        self._vs = self._ve = self._ps = self._pe = None

    def _now(self):
        if self.sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def attach(self, model):
        text_model = model.model.language_model
        visual = model.model.visual
        first_layer = text_model.layers[0]
        final_norm = text_model.norm

        def vis_pre(_m, _inp):
            if self._vs is None:
                self._vs = self._now()

        def vis_post(_m, _inp, _out):
            if self._ve is None:
                self._ve = self._now()

        def layer_pre(_m, _inp):
            if self._ps is None:
                self._ps = self._now()

        def norm_post(_m, _inp, _out):
            if self._pe is None and self._ps is not None:
                self._pe = self._now()

        self._hooks = [
            visual.register_forward_pre_hook(vis_pre),
            visual.register_forward_hook(vis_post),
            first_layer.register_forward_pre_hook(layer_pre),
            final_norm.register_forward_hook(norm_post),
        ]

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def _ms(a, b):
        return round((b - a) * 1000.0, 3) if a is not None and b is not None else None

    @property
    def vision_ms(self):
        return self._ms(self._vs, self._ve)

    @property
    def prefill_ms(self):
        return self._ms(self._ps, self._pe)


class RandomPruner(torch.nn.Module):
    """Stateless uniform-random selector with the same call signature as QueryAwarePruner."""
    def __init__(self):
        super().__init__()
        self._dtype_anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x_v, x_t, keep_ratio=0.5):
        B, N, _ = x_v.shape
        k = max(1, int(N * keep_ratio))
        idx = torch.stack([torch.randperm(N, device=x_v.device)[:k] for _ in range(B)], dim=0)
        idx, _ = torch.sort(idx, dim=1)
        return x_v.gather(1, idx.unsqueeze(-1).expand(-1, -1, x_v.shape[-1])), idx


def build_question(row) -> str:
    choices = "\n".join(row["multi-choice options"])
    return (
        "Look carefully at the image and answer the multiple-choice question.\n"
        "Choose the best option from A, B, C, D, and E.\n"
        "Respond with only the option letter, without explanation.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{choices}\n\n"
        "Answer:"
    )


def normalize_choice(s):
    if s is None:
        return None
    s = str(s).strip().upper()
    if s in {"A", "B", "C", "D", "E"}:
        return s
    m = _CHOICE_RE.match(s)
    return m.group(1) if m else None


def is_correct(pred_text, gt_letter):
    p, g = normalize_choice(pred_text), normalize_choice(gt_letter)
    return p is not None and g is not None and p == g


def _row_to_image(row):
    """The MME-RealWorld-lite parquet stores the JPEG as a base64 string in `bytes`."""
    cand = row["bytes"]
    if isinstance(cand, str):
        try:
            raw = base64.b64decode(cand, validate=False)
        except binascii.Error as e:
            raise ValueError(f"row {row.get('index')}: bad base64 in 'bytes' column: {e}") from e
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(cand, (bytes, bytearray)):
        return Image.open(io.BytesIO(cand)).convert("RGB")
    raise TypeError(f"row {row.get('index')}: unexpected 'bytes' type {type(cand).__name__}")


def load_samples(hf_dataset, split, categories, num_samples, seed, shuffle):
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    keep = set(categories)
    rows = [r for r in ds if r["category"] in keep]
    print(f"[data] {len(rows)} rows after filter on categories={sorted(keep)}")
    if shuffle:
        import random
        random.Random(seed).shuffle(rows)
    if num_samples > 0:
        rows = rows[:num_samples]
    out = []
    for row in rows:
        image = _row_to_image(row)
        task, _, subtask = row["category"].partition("/")
        out.append({
            "id": int(row["index"]),
            "image": image,
            "question_raw": row["question"],
            "question": build_question(row),
            "ground_truth": row["answer"],
            "task": task,
            "subtask": subtask,
            "l2_category": row.get("l2-category"),
            "category": row["category"],
        })
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
    }


def attach_pruner(model, args, device, dtype):
    text_cfg = getattr(model.config, "text_config", model.config)
    model_dim = text_cfg.hidden_size

    if args.pruner_type in ("none", "fastv"):
        return

    if args.pruner_type == "random":
        pruner = RandomPruner().to(device=device, dtype=dtype)
    else:  # learned
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
    model.keep_ratio = args.keep_ratio


def main():
    args = parse_args()
    device = torch.device(args.device)
    sync = device.type == "cuda"
    dtype = DTYPE[args.dtype]

    print(f"Loading {args.model_id} (dtype={args.dtype}, attn={args.attn_impl})")
    ModelCls = (
        FastVQwen2_5_VLForConditionalGeneration
        if args.pruner_type == "fastv"
        else PrunableQwen2_5_VLForConditionalGeneration
    )
    model = ModelCls.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_impl,
    )
    if args.pruner_type == "fastv":
        model.fastv_k = args.fastv_k
        model.fastv_r = 1.0 - args.keep_ratio
    model.eval()

    proc_kwargs = {}
    if args.min_pixels is not None:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    print(f"Processor pixel bounds: {proc_kwargs}")

    attach_pruner(model, args, device, dtype)

    stage_timer = StageTimer(sync_cuda=sync)
    stage_timer.attach(model)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    samples = load_samples(
        args.hf_dataset, args.hf_split, categories,
        args.num_samples, args.seed, args.shuffle,
    )
    print(f"Loaded {len(samples)} samples | mode={args.pruner_type} | keep_ratio={args.keep_ratio}")

    if args.warmup > 0 and samples:
        print(f"Warmup ({args.warmup} iters)...")
        warm = prepare_inputs(processor, model, samples[0]["image"], samples[0]["question"])
        warm_kwargs = {k: v for k, v in warm.items()
                       if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        with torch.no_grad():
            for _ in range(args.warmup):
                model.generate(**warm_kwargs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        if sync:
            torch.cuda.synchronize()

    results: List[Dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        try:
            stage_timer.reset()
            inputs = prepare_inputs(processor, model, s["image"], s["question"])
            input_len = int(inputs["input_ids"].shape[1])
            num_image_tokens = int((inputs["input_ids"][0] == model.config.image_token_id).sum().item())
            grid_thw = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None

            gen_kwargs = {k: v for k, v in inputs.items()
                          if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
            out_ids, times, t0, t_end = time_generate(model, gen_kwargs, args.max_new_tokens, sync)

            timing = summarize_timing(t0, times, t_end) or {}
            vision_ms = stage_timer.vision_ms
            prefill_ms = stage_timer.prefill_ms
            stats = getattr(model, "last_pruning_stats", None)
            if args.pruner_type != "none" and stats is not None:
                kept = stats["pruned_n"]
                original = stats["original_n"]
            else:
                kept = num_image_tokens
                original = num_image_tokens

            input_id_len = int(gen_kwargs["input_ids"].shape[1])
            gen_only = out_ids[0][input_id_len:]
            text = processor.tokenizer.decode(gen_only, skip_special_tokens=True)

            pred_letter = normalize_choice(text)
            gt_letter = normalize_choice(s["ground_truth"])
            correct = int(is_correct(text, s["ground_truth"]))

            row: Dict[str, Any] = {
                "id": s["id"],
                "task": s["task"],
                "subtask": s["subtask"],
                "l2_category": s["l2_category"],
                "category": s["category"],
                "question": s["question_raw"],
                "ground_truth": gt_letter,
                "prediction_raw": text,
                "prediction_letter": pred_letter,
                "correct": correct,
                "input_len": input_len,
                "num_image_tokens": num_image_tokens,
                "kept_image_tokens": kept,
                "tokens_pruned": original - kept,
                "image_grid_thw": grid_thw,
                "pruner_type": args.pruner_type,
                "keep_ratio": args.keep_ratio if args.pruner_type != "none" else None,
                "vision_ms": vision_ms,
                "prefill_ms": prefill_ms,
                "ttft_ms": timing.get("ttft_ms"),
                "avg_tbt_ms": timing.get("avg_tbt_ms"),
                "total_latency_ms": timing.get("total_latency_ms"),
                "num_decode_steps": timing.get("num_decode_steps"),
                "generated_token_count": int(gen_only.numel()),
            }
            results.append(row)
            print(
                f"[{i}/{len(samples)}] id={row['id']} | "
                f"gt={gt_letter} pred={pred_letter} ok={correct} | "
                f"img_tok={original}->{kept} | "
                f"vision={vision_ms}ms prefill={prefill_ms}ms "
                f"ttft={row['ttft_ms']}ms tbt={row['avg_tbt_ms']}ms total={row['total_latency_ms']}ms"
            )
        except Exception as e:
            print(f"  [{i}] error: {e}")
            results.append({"id": s.get("id"), "error": str(e)})

    ok = [r for r in results if "ttft_ms" in r and r["ttft_ms"] is not None]
    scored = [r for r in ok if r.get("prediction_letter") is not None]

    def acc_for(subset):
        return round(sum(r["correct"] for r in subset) / len(subset), 4) if subset else None

    by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok:
        by_subtask.setdefault(r["subtask"], []).append(r)

    summary = {
        "num_samples": len(samples),
        "num_ok": len(ok),
        "num_scored": len(scored),
        "accuracy_overall": acc_for(ok),
        "accuracy_with_valid_letter": acc_for(scored),
        "accuracy_by_subtask": {st: acc_for(rs) for st, rs in sorted(by_subtask.items())},
        "n_by_subtask": {st: len(rs) for st, rs in sorted(by_subtask.items())},
        "avg_vision_ms": round(
            mean(r["vision_ms"] for r in ok if r.get("vision_ms") is not None), 3
        ) if any(r.get("vision_ms") is not None for r in ok) else None,
        "avg_prefill_ms": round(
            mean(r["prefill_ms"] for r in ok if r.get("prefill_ms") is not None), 3
        ) if any(r.get("prefill_ms") is not None for r in ok) else None,
        "avg_ttft_ms": round(mean(r["ttft_ms"] for r in ok), 3) if ok else None,
        "avg_tbt_ms": round(
            mean(r["avg_tbt_ms"] for r in ok if r.get("avg_tbt_ms") is not None), 3
        ) if ok else None,
        "avg_total_latency_ms": round(mean(r["total_latency_ms"] for r in ok), 3) if ok else None,
        "avg_num_image_tokens": round(mean(r["num_image_tokens"] for r in ok), 2) if ok else None,
        "avg_kept_image_tokens": round(mean(r["kept_image_tokens"] for r in ok), 2) if ok else None,
        "avg_tokens_pruned": round(mean(r["tokens_pruned"] for r in ok), 2) if ok else None,
    }

    payload = {
        "config": {
            "model_id": args.model_id,
            "hf_dataset": args.hf_dataset,
            "hf_split": args.hf_split,
            "categories": categories,
            "pruner_type": args.pruner_type,
            "keep_ratio": args.keep_ratio if args.pruner_type != "none" else None,
            "fastv_k": args.fastv_k if args.pruner_type == "fastv" else None,
            "pruner_checkpoint": args.pruner_checkpoint or None,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "device": str(device),
            "warmup": args.warmup,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "num_samples_requested": args.num_samples,
            "seed": args.seed,
            "shuffle": args.shuffle,
        },
        "summary": summary,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
