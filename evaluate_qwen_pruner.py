"""Qwen2.5-VL pruned-generate evaluation pipeline.

Loads precomputed debiased importance maps, prunes vision tokens top-k
by importance, runs generate, and compares against the unpruned answer.

Two input layouts:

(A) Bundle mode -- recommended, matches the team's execution_outputs format:
    --bundle-dir <dir>           sample_<id>.pt is a self-contained dict:
                                 {sample_id, image_path, question, answer,
                                  tokens, debiased[, baseline]}
    --image-dir   <dir>          where sample_<id>.jpg images live
    --importance-key debiased    which key of the dict to use as importance

(B) Split mode -- the original layout from the minimal notebooks:
    --image-dir          *_<id>.jpg
    --qwen-outputs-dir   sample_<id>.pt with {question, model_answer}
    --importance-dir     {debiased,baseline}_importance_<id>.pt
    --importance-type    debiased | baseline

Output:
  --output JSON list, one record per (sample, layer, keep_ratio).

Bundle mode example:
  python evaluate_qwen_pruner.py \\
      --bundle-dir execution_outputs --image-dir images \\
      --sweep-layers 1,5,10,15,19 --sweep-keep-ratios 0.25,0.5,0.75 \\
      --limit 20 --output sweep.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", default=".", help="Directory containing sample_<id>.jpg or *_<id>.jpg images")
    # Bundle mode (recommended -- team's execution_outputs format)
    p.add_argument("--bundle-dir", default="",
                   help="Bundle mode: dir with sample_<id>.pt self-contained dicts "
                        "{question, answer, debiased, ...}. If set, overrides "
                        "--qwen-outputs-dir/--importance-dir/--importance-type.")
    p.add_argument("--importance-key", default="debiased",
                   help="Bundle mode: which key in the bundle dict holds the importance tensor "
                        "(e.g. 'debiased', 'baseline')")
    # Split mode (original layout)
    p.add_argument("--qwen-outputs-dir", default="qwen_outputs",
                   help="Split mode: dir with sample_<id>.pt bundles (question + model_answer)")
    p.add_argument("--importance-dir", default="importances_test",
                   help="Split mode: dir with <type>_importance_<id>.pt tensors")
    p.add_argument("--importance-type", default="debiased", choices=["debiased", "baseline"])
    p.add_argument("--layer", type=int, default=10, help="Importance layer index (single-config mode)")
    p.add_argument("--keep-ratio", type=float, default=0.5, help="Vision-token keep ratio (single-config mode)")
    p.add_argument("--reduce", default="mean", choices=["mean", "last", "max"],
                   help="How to collapse n_text dim of importance to a single [n_vision] vector")
    p.add_argument("--sweep-layers", default="",
                   help="Comma-separated layer indices, overrides --layer")
    p.add_argument("--sweep-keep-ratios", default="",
                   help="Comma-separated keep ratios, overrides --keep-ratio")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="Cap number of samples (0 = no cap)")
    p.add_argument("--output", default="eval_pruned_results.json")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Write partial results to --output every N records (0 = only at end). "
                        "Recommended for long runs to survive Colab disconnects.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    return p.parse_args()


def parse_id_from_stem(stem):
    """`anything_42` -> 42. Returns None if last segment isn't an int."""
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return None


def find_samples(image_dir, qwen_outputs_dir, importance_dir, importance_type):
    image_dir = Path(image_dir)
    qwen_outputs_dir = Path(qwen_outputs_dir)
    importance_dir = Path(importance_dir)

    images = {}
    for p in image_dir.glob("*.jpg"):
        sid = parse_id_from_stem(p.stem)
        if sid is not None:
            images[sid] = p

    bundles = {}
    for p in qwen_outputs_dir.glob("sample_*.pt"):
        sid = parse_id_from_stem(p.stem)
        if sid is not None:
            bundles[sid] = p

    importances = {}
    for p in importance_dir.glob(f"{importance_type}_importance_*.pt"):
        sid = parse_id_from_stem(p.stem)
        if sid is not None:
            importances[sid] = p

    common = sorted(set(images) & set(bundles) & set(importances))
    missing_img = sorted(set(bundles) & set(importances) - set(images))
    missing_imp = sorted(set(images) & set(bundles) - set(importances))
    if missing_img:
        print(f"[warn] {len(missing_img)} sample(s) missing image: {missing_img[:5]}{'...' if len(missing_img)>5 else ''}")
    if missing_imp:
        print(f"[warn] {len(missing_imp)} sample(s) missing importance: {missing_imp[:5]}{'...' if len(missing_imp)>5 else ''}")
    return [(sid, images[sid], bundles[sid], importances[sid]) for sid in common]


def find_samples_bundle(image_dir, bundle_dir):
    """Bundle mode: each sample_<id>.pt is self-contained (has question, answer, importance).
    Returns list of (sample_id, image_path, bundle_path)."""
    image_dir = Path(image_dir)
    bundle_dir = Path(bundle_dir)

    images = {}
    for p in image_dir.glob("*.jpg"):
        sid = parse_id_from_stem(p.stem)
        if sid is not None:
            images[sid] = p

    bundles = {}
    for p in bundle_dir.glob("sample_*.pt"):
        sid = parse_id_from_stem(p.stem)
        if sid is not None:
            bundles[sid] = p

    common = sorted(set(images) & set(bundles))
    missing_img = sorted(set(bundles) - set(images))
    missing_bundle = sorted(set(images) - set(bundles))
    if missing_img:
        print(f"[warn] {len(missing_img)} bundle(s) missing image: {missing_img[:5]}{'...' if len(missing_img)>5 else ''}")
    if missing_bundle:
        print(f"[warn] {len(missing_bundle)} image(s) missing bundle: {missing_bundle[:5]}{'...' if len(missing_bundle)>5 else ''}")
    return [(sid, images[sid], bundles[sid]) for sid in common]


def select_importance(importance_tensor, layer, reduce_method):
    """[num_layers, n_text, n_vision] -> [n_vision]"""
    if layer >= importance_tensor.shape[0]:
        raise IndexError(f"layer {layer} out of range (have {importance_tensor.shape[0]} layers)")
    layer_imp = importance_tensor[layer].float()
    if reduce_method == "mean":
        return layer_imp.mean(0)
    if reduce_method == "last":
        return layer_imp[-1]
    if reduce_method == "max":
        return layer_imp.max(0).values
    raise ValueError(f"unknown reduce: {reduce_method}")


def prepare_inputs(processor, model, image_path, question):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    return {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}


def vision_embeds(model, inputs):
    pv = inputs["pixel_values"]
    if hasattr(model.model.visual, "get_dtype"):
        pv = pv.type(model.model.visual.get_dtype())
    with torch.no_grad():
        embeds = model.model.visual(pv, grid_thw=inputs["image_grid_thw"])
    return embeds.pooler_output if hasattr(embeds, "pooler_output") else embeds


def build_pruned_inputs(model, inputs, vision_importance, keep_ratio):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_embeds = vision_embeds(model, inputs)
    image_positions = torch.where(input_ids[0] == model.config.image_token_id)[0]

    n_viz = image_embeds.shape[0]
    if n_viz != image_positions.numel():
        raise ValueError(f"image_embeds={n_viz} != image_positions={image_positions.numel()}")
    if vision_importance.shape[0] != n_viz:
        raise ValueError(f"importance has {vision_importance.shape[0]} entries but n_viz={n_viz}")

    k = max(1, round(n_viz * keep_ratio))
    keep_idx = torch.sort(torch.topk(vision_importance.to(model.device), k).indices).values

    keep_seq = torch.ones_like(input_ids[0], dtype=torch.bool, device=model.device)
    keep_seq[image_positions] = False
    keep_seq[image_positions[keep_idx]] = True

    text_embeds = model.model.language_model.embed_tokens(input_ids)
    new_ids = input_ids[:, keep_seq]
    new_embeds = text_embeds[:, keep_seq].clone()
    new_embeds[0, new_ids[0] == model.config.image_token_id] = image_embeds[keep_idx].to(new_embeds.dtype)

    position_ids, rope_deltas = model.model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=attention_mask,
        mm_token_type_ids=inputs["mm_token_type_ids"],
    )
    return {
        "inputs_embeds": new_embeds,
        "attention_mask": attention_mask[:, keep_seq],
        "position_ids": position_ids[:, :, keep_seq],
        "rope_deltas": rope_deltas,
        "n_viz_total": n_viz,
        "n_viz_kept": k,
    }


@torch.no_grad()
def generate_pruned(model, processor, pruned, max_new_tokens):
    out_ids = model.generate(
        inputs_embeds=pruned["inputs_embeds"],
        attention_mask=pruned["attention_mask"],
        position_ids=pruned["position_ids"],
        rope_deltas=pruned["rope_deltas"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return processor.batch_decode(out_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


def normalize_answer(s):
    if s is None:
        return ""
    return " ".join(s.lower().strip().split())


def parse_csv_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_csv_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()

    layers = parse_csv_ints(args.sweep_layers) if args.sweep_layers else [args.layer]
    keep_ratios = parse_csv_floats(args.sweep_keep_ratios) if args.sweep_keep_ratios else [args.keep_ratio]

    bundle_mode = bool(args.bundle_dir)
    if bundle_mode:
        samples = find_samples_bundle(args.image_dir, args.bundle_dir)
        importance_label = args.importance_key  # for record bookkeeping
    else:
        samples = find_samples(args.image_dir, args.qwen_outputs_dir,
                               args.importance_dir, args.importance_type)
        importance_label = args.importance_type

    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        if bundle_mode:
            raise SystemExit(
                "No samples matched (bundle mode). Need image+sample_<id>.pt for same id.\n"
                f"  image-dir: {args.image_dir}\n"
                f"  bundle-dir: {args.bundle_dir}"
            )
        raise SystemExit(
            "No samples matched. Need image+sample_<id>.pt+importance for the same sample id.\n"
            f"  image-dir: {args.image_dir}\n"
            f"  qwen-outputs-dir: {args.qwen_outputs_dir}\n"
            f"  importance-dir: {args.importance_dir} (type={args.importance_type})"
        )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"Loading {args.model_id} on {args.device} ({args.dtype}, attn={args.attn_impl})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=args.device, attn_implementation=args.attn_impl,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    n_runs = len(samples) * len(layers) * len(keep_ratios)
    print(f"Running {len(samples)} sample(s) x {len(layers)} layer(s) x {len(keep_ratios)} keep-ratio(s) = {n_runs} run(s)")

    results = []
    for s_idx, sample in enumerate(samples):
        if bundle_mode:
            sid, img_path, bundle_path = sample
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            question = bundle.get("question", "")
            # bundle uses "answer" (team format); fall back to "model_answer" for back-compat
            baseline_answer = bundle.get("answer", bundle.get("model_answer", None))
            if args.importance_key not in bundle:
                print(f"  [skip] sid={sid} bundle missing key '{args.importance_key}' "
                      f"(have: {list(bundle.keys())})")
                continue
            importance_tensor = bundle[args.importance_key]
        else:
            sid, img_path, bundle_path, imp_path = sample
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            question = bundle.get("question", "")
            baseline_answer = bundle.get("model_answer", None)
            importance_tensor = torch.load(imp_path, map_location="cpu", weights_only=False)

        inputs = prepare_inputs(processor, model, img_path, question)

        for layer in layers:
            if layer >= importance_tensor.shape[0]:
                print(f"  [skip] sid={sid} layer={layer} >= importance_layers={importance_tensor.shape[0]}")
                continue
            vision_importance = select_importance(importance_tensor, layer, args.reduce)

            for keep_ratio in keep_ratios:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pruned = build_pruned_inputs(model, inputs, vision_importance, keep_ratio)
                pruned_answer = generate_pruned(model, processor, pruned, args.max_new_tokens)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - t0

                rec = {
                    "sample_id": sid,
                    "question": question,
                    "baseline_answer": baseline_answer,
                    "pruned_answer": pruned_answer,
                    "exact_match": normalize_answer(baseline_answer) == normalize_answer(pruned_answer),
                    "keep_ratio": keep_ratio,
                    "layer": layer,
                    "reduce": args.reduce,
                    "importance_type": importance_label,
                    "n_viz_total": pruned["n_viz_total"],
                    "n_viz_kept": pruned["n_viz_kept"],
                    "latency_s": round(latency, 4),
                }
                results.append(rec)
                print(f"  [{s_idx+1}/{len(samples)}] sid={sid} layer={layer} keep={keep_ratio} "
                      f"viz {pruned['n_viz_total']}->{pruned['n_viz_kept']} "
                      f"em={rec['exact_match']} t={latency:.2f}s")

                if args.checkpoint_every > 0 and len(results) % args.checkpoint_every == 0:
                    Path(args.output).write_text(json.dumps(results, indent=2))
                    print(f"  [checkpoint] wrote {len(results)} records to {args.output}")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} record(s) to {out_path}")

    if results:
        em_overall = sum(r["exact_match"] for r in results) / len(results)
        print(f"\nOverall exact-match vs unpruned: {em_overall*100:.1f}%")
        by_config = defaultdict(list)
        for r in results:
            by_config[(r["layer"], r["keep_ratio"])].append(r)
        print("\nPer-config exact-match (layer, keep_ratio):")
        for (l, kr), rs in sorted(by_config.items()):
            em = sum(r["exact_match"] for r in rs) / len(rs)
            avg_lat = sum(r["latency_s"] for r in rs) / len(rs)
            print(f"  layer={l:3d} keep={kr:.2f}: em={em*100:5.1f}%  avg_latency={avg_lat:.2f}s  n={len(rs)}")


if __name__ == "__main__":
    main()
