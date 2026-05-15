"""Pruned Qwen MME-RealWorld-Lite Exact-Match Accuracy (Debiased Oracle).

Sibling of ``evaluate_pruned_accuracy_mme_exact_match.ipynb``. Same eval
setup (debiased-oracle top-k pruning at a fixed keep ratio, exact-match
accuracy against the gold letter), but adapted to the MME-RealWorld-Lite
camera-perception subset.

Run:
    cd mme_data
    python evaluate_pruned_accuracy_mme_lite_exact_match.py
"""
# %%
import base64
import io
import json
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# %% Config ---------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("mme_data")
IMPORTANCE_DIR = DATA_DIR / "lite_execution_outputs"
OUTPUT_PATH = DATA_DIR / "pruned_accuracy_lite_exact_match_results_0p2.jsonl"
SUMMARY_PATH = DATA_DIR / "pruned_accuracy_lite_exact_match_summary.json"

HF_DATASET = "yifanzhang114/MME-RealWorld-lite-lmms-eval"
HF_SPLIT = "train"
CAMERA_CATEGORIES = {"Perception/Autonomous_Driving", "Perception/Monitoring"}

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
KEEP_RATIO = 0.2
IMPORTANCE_LAYER = 10  # Used only if a saved importance bundle still has a layer dimension.
MAX_NEW_TOKENS = 8

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

START_SAMPLE_ID = None  # e.g. 100 starts at sample_id 100; None = first row
LIMIT = None  # None = full 669-row sweep

assert 0 < KEEP_RATIO <= 1, "KEEP_RATIO must be in (0, 1]"

# %% Helpers --------------------------------------------------------------

_CHOICE_RE = re.compile(r"^\(?\s*([ABCDE])\s*\)?(?:\s|\.|,|:|$)")


def _row_to_image(row):
    """The MME-RealWorld-lite parquet stores the JPEG as base64 in 'bytes'."""
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


def build_importance_index():
    if not IMPORTANCE_DIR.exists():
        return {}
    paths = {}
    for path in IMPORTANCE_DIR.glob("sample_*.pt"):
        sample_id = int(path.stem.rsplit("_", 1)[-1])
        paths[sample_id] = path
    return paths


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


def load_vision_importance(path, layer):
    bundle = torch.load(path, map_location="cpu")
    importance = bundle["debiased"] if isinstance(bundle, dict) and "debiased" in bundle else bundle
    if importance.ndim == 3:
        if layer < 0 or layer >= importance.shape[0]:
            raise ValueError(f"IMPORTANCE_LAYER {layer} is outside tensor shape {tuple(importance.shape)}")
        importance = importance[layer].mean(0)
    elif importance.ndim == 2:
        importance = importance.mean(0)
    elif importance.ndim != 1:
        raise ValueError(f"Expected 1D, 2D, or 3D importance tensor, got shape {tuple(importance.shape)}")
    if isinstance(bundle, dict) and "num_image_tokens" in bundle and importance.numel() != int(bundle["num_image_tokens"]):
        raise ValueError(f"{path} has {importance.numel()} importance scores but {bundle['num_image_tokens']} image tokens")
    return importance.float()


def build_pruned_inputs(model, inputs, vision_importance, keep_ratio):
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    image_embeds = vision_embeds(model, inputs)
    image_positions = torch.where(input_ids[0] == model.config.image_token_id)[0]
    if image_embeds.shape[0] != image_positions.numel():
        raise ValueError(f"image_embeds={image_embeds.shape[0]} image_positions={image_positions.numel()}")
    if vision_importance.numel() != image_positions.numel():
        raise ValueError(f"importance={vision_importance.numel()} image_positions={image_positions.numel()}")

    k = max(1, round(image_embeds.shape[0] * keep_ratio))
    keep_idx = torch.sort(torch.topk(vision_importance.to(model.device), k).indices).values
    keep_seq = torch.ones_like(input_ids[0], dtype=torch.bool, device=model.device)
    keep_seq[image_positions] = False
    keep_seq[image_positions[keep_idx]] = True

    text_embeds = model.model.language_model.embed_tokens(input_ids)
    new_ids = input_ids[:, keep_seq]
    new_embeds = text_embeds[:, keep_seq].clone()
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


def run_unpruned(processor, model, inputs, max_new_tokens):
    """Generate the no-pruning baseline answer for the current sample."""
    gen_kwargs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    with torch.no_grad():
        ids = model.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    input_len = inputs["input_ids"].shape[1]
    gen_only = ids[0][input_len:]
    return processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()


def run_pruned(processor, model, inputs, vision_importance, keep_ratio, max_new_tokens):
    pruned = build_pruned_inputs(model, inputs, vision_importance, keep_ratio)
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
    with torch.no_grad():
        ids = model.generate(**gen_kwargs)
    answer = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer.strip(), pruned


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


# %% Main -----------------------------------------------------------------

def main():
    samples = load_samples()
    if START_SAMPLE_ID is not None:
        samples = [s for s in samples if s["sample_id"] >= int(START_SAMPLE_ID)]
    if LIMIT is not None:
        samples = samples[:LIMIT]
    if not samples:
        raise ValueError("No samples after filtering.")

    importance_paths = build_importance_index()
    print(f"qwen model: {MODEL_NAME}")
    print(f"samples: {len(samples)}")
    print(f"importance files in {IMPORTANCE_DIR}: {len(importance_paths)}")
    print(f"keep_ratio: {KEEP_RATIO}")
    print(f"first sample: id={samples[0]['sample_id']} gt={samples[0]['ground_truth_answer']}")
    if not importance_paths:
        raise FileNotFoundError(
            f"No importance bundles found under {IMPORTANCE_DIR}. "
            "Generate them with the offline provenance pipeline on the lite "
            "samples first (parallel to mme_data/execution_outputs/ for the "
            "original 500-sample subset)."
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    print(torch.__version__, torch.version.cuda, torch.cuda.is_available())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    totals = {"num_samples": 0, "unpruned_correct": 0, "pruned_correct": 0}
    unpruned_correct_pruned_incorrect_ids = []
    i = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for sample in samples:
            sample_id = sample["sample_id"]
            importance_path = importance_paths.get(sample_id)
            if importance_path is None:
                raise FileNotFoundError(
                    f"No importance file for sample_id={sample_id}. "
                    f"Expected {IMPORTANCE_DIR}/sample_{sample_id}.pt"
                )

            inputs = prepare_inputs(processor, model, sample["image"], sample["question"])
            unpruned_answer = run_unpruned(processor, model, inputs, MAX_NEW_TOKENS)
            vision_importance = load_vision_importance(importance_path, IMPORTANCE_LAYER)
            pruned_answer, pruned = run_pruned(
                processor, model, inputs, vision_importance, KEEP_RATIO, MAX_NEW_TOKENS,
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
                "importance_layer": IMPORTANCE_LAYER,
                "num_image_tokens": pruned["num_image_tokens"],
                "num_kept_image_tokens": pruned["num_kept_image_tokens"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            totals["num_samples"] += 1
            totals["unpruned_correct"] += int(unpruned_correct)
            totals["pruned_correct"] += int(pruned_correct)
            if unpruned_correct and not pruned_correct:
                unpruned_correct_pruned_incorrect_ids.append(sample_id)

            if i % 10 == 0:
                u = totals["unpruned_correct"] / max(1, totals["num_samples"])
                p = totals["pruned_correct"] / max(1, totals["num_samples"])
                print(
                    f"i={i} sample_id={sample_id} "
                    f"kept={row['num_kept_image_tokens']}/{row['num_image_tokens']} "
                    f"unpruned={row['normalized_unpruned_answer']} "
                    f"pruned={row['normalized_pruned_answer']} "
                    f"gt={row['normalized_ground_truth_answer']} | "
                    f"unpruned_acc={u:.4f} pruned_acc={p:.4f}"
                )
            i += 1

    summary = {
        **totals,
        "unpruned_accuracy": totals["unpruned_correct"] / max(1, totals["num_samples"]),
        "pruned_accuracy": totals["pruned_correct"] / max(1, totals["num_samples"]),
        "keep_ratio": KEEP_RATIO,
        "importance_layer": IMPORTANCE_LAYER,
        "categories": sorted(CAMERA_CATEGORIES),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "unpruned_correct_pruned_incorrect_ids": unpruned_correct_pruned_incorrect_ids,
        "hf_dataset": HF_DATASET,
        "results_path": str(OUTPUT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
