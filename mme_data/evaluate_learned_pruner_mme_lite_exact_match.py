"""Learned-Pruner Qwen MME-RealWorld-Lite Exact-Match Accuracy + Latency.

Sibling of ``evaluate_learned_pruner_mme_exact_match.ipynb``. Same eval
setup (learned-pruner top-k pruning at a fixed keep ratio, exact-match
accuracy against the gold letter, ``best_pruner_mme.pt`` checkpoint), but
adapted to the MME-RealWorld-Lite camera-perception subset and extended
with per-phase latency instrumentation.

Latency measured per sample (CUDA-synced):
  * ``vision_ms``  : ViT forward (via hook on ``model.model.visual``).
  * ``pruner_ms``  : learned-pruner forward (pruned path only).
  * ``prefill_ms`` : LLM-only prefill, via hooks on ``text_model.layers[0]``
                    pre and ``text_model.norm`` post — i.e. the decoder
                    forward, excluding vision, embed-merge, and lm_head.
  * ``ttft_ms``    : ``t0 -> first logit``. ``t0`` is taken BEFORE
                    ``vision_embeds(...)`` in the pruned path so vision
                    and pruner time are included, matching the unpruned
                    convention (where vision runs inside ``generate``).
  * ``avg_tbt_ms`` : mean inter-decode-step gap.
  * ``total_latency_ms`` : ``t_end - t0`` around the full generation.

Differences from the original notebook:
  * Samples are pulled directly from HF
    (``yifanzhang114/MME-RealWorld-lite-lmms-eval``) and filtered to
    ``Perception/Autonomous_Driving`` and ``Perception/Monitoring``
    (669 rows). Image bytes are base64-encoded JPEG strings in the
    parquet ``bytes`` column and are decoded inline.
  * No pre-generated unpruned manifest: both unpruned and pruned answers
    are generated per sample and scored against the gold letter.
  * Processor pixel bounds are hard-coded to ``min=256*28*28`` and
    ``max=1024*28*28`` (matches the lite eval setup and the pruner's
    training resolution).
  * A 2-iter warmup is run before the main loop so the first sample's
    latency is not inflated by JIT/kernel compilation.
"""
# %%
import base64
import io
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen2_5_VLForConditionalGeneration

# %% Config ---------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("mme_data")
PRUNER_CHECKPOINT = DATA_DIR / "best_pruner_mme.pt"
OUTPUT_PATH = DATA_DIR / "learned_pruner_lite_exact_match_results_0p2.jsonl"
SUMMARY_PATH = DATA_DIR / "learned_pruner_lite_exact_match_summary.json"

HF_DATASET = "yifanzhang114/MME-RealWorld-lite-lmms-eval"
HF_SPLIT = "train"
CAMERA_CATEGORIES = {"Perception/Autonomous_Driving", "Perception/Monitoring"}

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
KEEP_RATIO = 0.2
MAX_NEW_TOKENS = 8

# Match the lite eval setup used elsewhere and the pruner's training resolution.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

WARMUP = 2

START_SAMPLE_ID = None
LIMIT = None

assert 0 < KEEP_RATIO <= 1, "KEEP_RATIO must be in (0, 1]"

# %% Helpers --------------------------------------------------------------

_CHOICE_RE = re.compile(r"^\(?\s*([ABCDE])\s*\)?(?:\s|\.|,|:|$)")


def _row_to_image(row):
    cand = row["bytes"]
    if isinstance(cand, str):
        raw = base64.b64decode(cand, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(cand, (bytes, bytearray)):
        return Image.open(io.BytesIO(cand)).convert("RGB")
    raise TypeError(f"row {row.get('index')}: unexpected 'bytes' type {type(cand).__name__}")


def build_question(row):
    choices = "\n".join(row["multi-choice options"])
    return (
        "Look carefully at the image and answer the multiple-choice question.\n"
        "Choose the best option from A, B, C, D, and E.\n"
        "Respond with only the option letter, without explanation.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{choices}\n\n"
        "Answer:"
    )


def load_samples():
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    rows = [r for r in ds if r["category"] in CAMERA_CATEGORIES]
    out = []
    for r in rows:
        task, _, subtask = r["category"].partition("/")
        out.append({
            "sample_id": int(r["index"]),
            "question": build_question(r),
            "question_raw": r["question"],
            "ground_truth_answer": r["answer"],
            "category": r["category"],
            "task": task,
            "subtask": subtask,
            "image": _row_to_image(r),
        })
    return out


def prepare_inputs(processor, model, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}


def vision_embeds(model, inputs):
    dtype = model.model.visual.get_dtype() if hasattr(model.model.visual, "get_dtype") else inputs["pixel_values"].dtype
    pixel_values = inputs["pixel_values"].type(dtype)
    with torch.no_grad():
        embeds = model.model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
    return embeds.pooler_output if hasattr(embeds, "pooler_output") else embeds


def text_embeds(model, inputs):
    return model.model.language_model.embed_tokens(inputs["input_ids"]).to(torch.float32)


class QueryAwarePruner(nn.Module):
    def __init__(self, dim, num_heads=8, use_multi_head=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_multi_head = use_multi_head
        self.head_dim = dim // num_heads
        self.last_importance_scores = None
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        if use_multi_head:
            self.head_merger = nn.Linear(num_heads, 1)
        self.mlp_scorer = nn.Sequential(
            nn.Linear(dim + 1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def get_normal_attention(self, x_v, x_t):
        q = self.q_proj(x_t)
        k = self.k_proj(x_v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dim ** 0.5)
        return scores.max(dim=1).values.unsqueeze(-1)

    def get_multi_head_attention(self, x_v, x_t):
        batch_size, text_len, _ = x_t.shape
        num_vision = x_v.shape[1]
        q = self.q_proj(x_t).view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_v).view(batch_size, num_vision, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        head_relevance = scores.max(dim=-2).values.transpose(1, 2)
        return self.head_merger(head_relevance)

    def forward(self, x_v, x_t, keep_ratio=0.5):
        if self.use_multi_head:
            visual_relevance = self.get_multi_head_attention(x_v, x_t)
        else:
            visual_relevance = self.get_normal_attention(x_v, x_t)

        combined = torch.cat([x_v, visual_relevance], dim=-1)
        importance = self.mlp_scorer(combined).squeeze(-1)
        self.last_importance_scores = importance

        if self.training:
            return x_v * importance.unsqueeze(-1), None

        keep_k = max(1, int(x_v.shape[1] * keep_ratio))
        _, keep_idx = torch.topk(importance, keep_k, dim=1)
        keep_idx, _ = torch.sort(keep_idx, dim=1)
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, self.dim)
        pruned_v = torch.gather(x_v, 1, gather_idx)
        return pruned_v, keep_idx


def load_pruner(checkpoint_path, model):
    text_cfg = getattr(model.config, "text_config", model.config)
    model_dim = text_cfg.hidden_size
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("pruner_config", {"dim": model_dim})
    if cfg.get("dim", model_dim) != model_dim:
        raise ValueError(f"pruner dim {cfg.get('dim')} != Qwen hidden size {model_dim}")

    pruner = QueryAwarePruner(**cfg).to(device=model.device, dtype=torch.float32)
    pruner.load_state_dict(ckpt.get("state_dict", ckpt))
    pruner.eval()
    return pruner, ckpt


def score_with_pruner(pruner, image_embeds, text_embeds):
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(0)
    with torch.no_grad():
        pruner(image_embeds.to(torch.float32), text_embeds.to(torch.float32), keep_ratio=1.0)
        scores = pruner.last_importance_scores.squeeze(0)
    if scores.numel() != image_embeds.shape[1]:
        raise ValueError(f"Expected {image_embeds.shape[1]} scores, got {scores.numel()}")
    return scores


def build_pruned_inputs(model, inputs, image_embeds, vision_scores, keep_ratio):
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    image_positions = torch.where(input_ids[0] == model.config.image_token_id)[0]
    if image_embeds.shape[0] != image_positions.numel():
        raise ValueError(f"image_embeds={image_embeds.shape[0]} image_positions={image_positions.numel()}")
    if vision_scores.numel() != image_positions.numel():
        raise ValueError(f"scores={vision_scores.numel()} image_positions={image_positions.numel()}")

    k = max(1, int(image_embeds.shape[0] * keep_ratio))
    keep_idx = torch.sort(torch.topk(vision_scores.to(model.device), k).indices).values
    keep_seq = torch.ones_like(input_ids[0], dtype=torch.bool, device=model.device)
    keep_seq[image_positions] = False
    keep_seq[image_positions[keep_idx]] = True

    token_embeds = model.model.language_model.embed_tokens(input_ids)
    new_ids = input_ids[:, keep_seq]
    new_embeds = token_embeds[:, keep_seq].clone()
    new_embeds[0, new_ids[0] == model.config.image_token_id] = image_embeds[keep_idx].to(new_embeds.dtype)

    rope_kwargs = {
        "input_ids": input_ids,
        "image_grid_thw": inputs["image_grid_thw"],
        "attention_mask": attention_mask,
    }
    if "mm_token_type_ids" in inputs:
        rope_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"]
    position_ids, rope_deltas = model.model.get_rope_index(**rope_kwargs)

    pruned = {
        "inputs_embeds": new_embeds,
        "attention_mask": attention_mask[:, keep_seq],
        "position_ids": position_ids[:, :, keep_seq],
        "rope_deltas": rope_deltas,
        "num_image_tokens": int(image_positions.numel()),
        "num_kept_image_tokens": int(k),
    }
    if "mm_token_type_ids" in inputs:
        pruned["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, keep_seq]
    return pruned


def normalize_choice(answer):
    text = str(answer).strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    m = _CHOICE_RE.match(text)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot normalize choice from: {answer!r}")


def exact_match_choice(model_answer, ground_truth_answer):
    return normalize_choice(model_answer) == normalize_choice(ground_truth_answer)


# %% Latency instrumentation ----------------------------------------------

class TimingLogitsProcessor(LogitsProcessor):
    def __init__(self, sync_cuda: bool):
        self.times: List[float] = []
        self.sync = sync_cuda

    def __call__(self, input_ids, scores):
        if self.sync:
            torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        return scores


class StageTimer:
    """Captures vision-tower and LLM-decoder forward times via module hooks.

    Hooks only record the FIRST call after each ``reset()`` so decode-step
    invocations of layers[0] do not overwrite the prefill measurement.
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

    @staticmethod
    def _ms(a, b):
        return round((b - a) * 1000.0, 3) if a is not None and b is not None else None

    @property
    def vision_ms(self):
        return self._ms(self._vs, self._ve)

    @property
    def prefill_ms(self):
        return self._ms(self._ps, self._pe)


def _summarize_decode(t0, times, t_end):
    if not times:
        return {"ttft_ms": None, "avg_tbt_ms": None, "total_latency_ms": None, "num_decode_steps": 0}
    inter = [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)]
    return {
        "ttft_ms": round((times[0] - t0) * 1000.0, 3),
        "avg_tbt_ms": round(mean(inter), 3) if inter else None,
        "total_latency_ms": round((t_end - t0) * 1000.0, 3),
        "num_decode_steps": len(times),
    }


def _sync(sync):
    if sync:
        torch.cuda.synchronize()


def run_unpruned(processor, model, inputs, max_new_tokens, stage_timer, sync):
    """Generate the no-pruning baseline answer and record per-phase latency."""
    stage_timer.reset()
    timing = TimingLogitsProcessor(sync)
    gen_kwargs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    _sync(sync)
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            logits_processor=LogitsProcessorList([timing]),
        )
    _sync(sync)
    t_end = time.perf_counter()

    input_len = inputs["input_ids"].shape[1]
    gen_only = ids[0][input_len:]
    answer = processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    latency = {
        "vision_ms": stage_timer.vision_ms,
        "pruner_ms": None,
        "prefill_ms": stage_timer.prefill_ms,
        **_summarize_decode(t0, timing.times, t_end),
    }
    return answer, latency


def run_pruned(processor, model, pruner, inputs, keep_ratio, max_new_tokens, stage_timer, sync):
    """Run the full pruned path and record per-phase latency.

    ``t0`` is taken before ``vision_embeds(...)`` so the reported TTFT is
    end-to-end (vision + pruner + LLM prefill), matching the convention
    used in the unpruned path where vision runs inside ``generate``.
    """
    stage_timer.reset()
    _sync(sync)
    t0 = time.perf_counter()

    image_features = vision_embeds(model, inputs)
    query_features = text_embeds(model, inputs)

    _sync(sync)
    t_pruner_start = time.perf_counter()
    vision_scores = score_with_pruner(pruner, image_features, query_features)
    _sync(sync)
    pruner_ms = round((time.perf_counter() - t_pruner_start) * 1000.0, 3)

    pruned = build_pruned_inputs(model, inputs, image_features, vision_scores, keep_ratio)
    gen_kwargs = {
        "inputs_embeds": pruned["inputs_embeds"],
        "attention_mask": pruned["attention_mask"],
        "position_ids": pruned["position_ids"],
        "rope_deltas": pruned["rope_deltas"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if "mm_token_type_ids" in pruned:
        gen_kwargs["mm_token_type_ids"] = pruned["mm_token_type_ids"]

    timing = TimingLogitsProcessor(sync)
    with torch.no_grad():
        ids = model.generate(**gen_kwargs, logits_processor=LogitsProcessorList([timing]))
    _sync(sync)
    t_end = time.perf_counter()

    answer = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    latency = {
        "vision_ms": stage_timer.vision_ms,
        "pruner_ms": pruner_ms,
        "prefill_ms": stage_timer.prefill_ms,
        **_summarize_decode(t0, timing.times, t_end),
    }
    return answer, pruned, latency


# %% Main -----------------------------------------------------------------

def main():
    samples = load_samples()
    if START_SAMPLE_ID is not None:
        samples = [s for s in samples if s["sample_id"] >= int(START_SAMPLE_ID)]
    if LIMIT is not None:
        samples = samples[:LIMIT]
    if not samples:
        raise ValueError("No samples after filtering.")

    print(f"qwen model: {MODEL_NAME}")
    print(f"samples: {len(samples)}")
    print(f"pruner checkpoint: {PRUNER_CHECKPOINT}")
    print(f"keep_ratio: {KEEP_RATIO}")
    print(f"first sample: id={samples[0]['sample_id']} gt={samples[0]['ground_truth_answer']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    print(torch.__version__, torch.version.cuda, torch.cuda.is_available())

    pruner, pruner_ckpt = load_pruner(PRUNER_CHECKPOINT, model)
    print(f"loaded pruner config: {pruner_ckpt.get('pruner_config')}")

    sync = device.type == "cuda"
    stage_timer = StageTimer(sync_cuda=sync)
    stage_timer.attach(model)

    if WARMUP > 0 and samples:
        print(f"Warmup ({WARMUP} iters)...")
        warm_inputs = prepare_inputs(processor, model, samples[0]["image"], samples[0]["question"])
        for _ in range(WARMUP):
            run_unpruned(processor, model, warm_inputs, MAX_NEW_TOKENS, stage_timer, sync)
            run_pruned(processor, model, pruner, warm_inputs, KEEP_RATIO, MAX_NEW_TOKENS, stage_timer, sync)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    totals = {"num_samples": 0, "unpruned_correct": 0, "pruned_correct": 0}
    unpruned_correct_pruned_incorrect_ids = []
    latency_rows: List[Dict[str, Optional[float]]] = []
    i = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for sample in samples:
            sample_id = sample["sample_id"]
            inputs = prepare_inputs(processor, model, sample["image"], sample["question"])
            unpruned_answer, unpruned_lat = run_unpruned(
                processor, model, inputs, MAX_NEW_TOKENS, stage_timer, sync,
            )
            pruned_answer, pruned, pruned_lat = run_pruned(
                processor, model, pruner, inputs, KEEP_RATIO, MAX_NEW_TOKENS, stage_timer, sync,
            )
            try:
                unpruned_correct = exact_match_choice(unpruned_answer, sample["ground_truth_answer"])
                pruned_correct = exact_match_choice(pruned_answer, sample["ground_truth_answer"])
            except ValueError:
                print(f"i={i} sample_id={sample_id}: parsing failed, "
                      f"unpruned={unpruned_answer!r} pruned={pruned_answer!r}")
                i += 1
                continue

            row = {
                "sample_id": sample_id,
                "task": sample["task"],
                "subtask": sample["subtask"],
                "category": sample["category"],
                "question": sample["question_raw"],
                "ground_truth_answer": sample["ground_truth_answer"],
                "unpruned_answer": unpruned_answer,
                "pruned_answer": pruned_answer,
                "normalized_unpruned_answer": normalize_choice(unpruned_answer),
                "normalized_pruned_answer": normalize_choice(pruned_answer),
                "normalized_ground_truth_answer": normalize_choice(sample["ground_truth_answer"]),
                "unpruned_correct": unpruned_correct,
                "pruned_correct": pruned_correct,
                "keep_ratio": KEEP_RATIO,
                "pruner_checkpoint": str(PRUNER_CHECKPOINT),
                "num_image_tokens": pruned["num_image_tokens"],
                "num_kept_image_tokens": pruned["num_kept_image_tokens"],
                "unpruned_latency": unpruned_lat,
                "pruned_latency": pruned_lat,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            latency_rows.append({"unpruned": unpruned_lat, "pruned": pruned_lat})
            totals["num_samples"] += 1
            totals["unpruned_correct"] += int(unpruned_correct)
            totals["pruned_correct"] += int(pruned_correct)
            if unpruned_correct and not pruned_correct:
                unpruned_correct_pruned_incorrect_ids.append(sample_id)

            if i % 20 == 0:
                u = totals["unpruned_correct"] / max(1, totals["num_samples"])
                p = totals["pruned_correct"] / max(1, totals["num_samples"])
                print(
                    f"i={i} id={sample_id} kept={row['num_kept_image_tokens']}/{row['num_image_tokens']} | "
                    f"u={row['normalized_unpruned_answer']} p={row['normalized_pruned_answer']} "
                    f"gt={row['normalized_ground_truth_answer']} | u_acc={u:.4f} p_acc={p:.4f} | "
                    f"u_prefill={unpruned_lat['prefill_ms']}ms p_prefill={pruned_lat['prefill_ms']}ms "
                    f"pruner={pruned_lat['pruner_ms']}ms"
                )
            i += 1

    def _avg(rows, side, key):
        vs = [r[side][key] for r in rows if r[side].get(key) is not None]
        return round(mean(vs), 3) if vs else None

    summary = {
        **totals,
        "unpruned_accuracy": totals["unpruned_correct"] / max(1, totals["num_samples"]),
        "pruned_accuracy": totals["pruned_correct"] / max(1, totals["num_samples"]),
        "keep_ratio": KEEP_RATIO,
        "pruner_checkpoint": str(PRUNER_CHECKPOINT),
        "pruner_config": pruner_ckpt.get("pruner_config"),
        "categories": sorted(CAMERA_CATEGORIES),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "hf_dataset": HF_DATASET,
        "results_path": str(OUTPUT_PATH),
        "unpruned_correct_pruned_incorrect_ids": unpruned_correct_pruned_incorrect_ids,
        "avg_latency_unpruned": {
            "vision_ms": _avg(latency_rows, "unpruned", "vision_ms"),
            "prefill_ms": _avg(latency_rows, "unpruned", "prefill_ms"),
            "ttft_ms": _avg(latency_rows, "unpruned", "ttft_ms"),
            "avg_tbt_ms": _avg(latency_rows, "unpruned", "avg_tbt_ms"),
            "total_latency_ms": _avg(latency_rows, "unpruned", "total_latency_ms"),
        },
        "avg_latency_pruned": {
            "vision_ms": _avg(latency_rows, "pruned", "vision_ms"),
            "pruner_ms": _avg(latency_rows, "pruned", "pruner_ms"),
            "prefill_ms": _avg(latency_rows, "pruned", "prefill_ms"),
            "ttft_ms": _avg(latency_rows, "pruned", "ttft_ms"),
            "avg_tbt_ms": _avg(latency_rows, "pruned", "avg_tbt_ms"),
            "total_latency_ms": _avg(latency_rows, "pruned", "total_latency_ms"),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
