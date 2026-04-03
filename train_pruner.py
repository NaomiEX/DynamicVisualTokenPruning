import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

from custom_llava_new import PrunableLlavaForConditionalGeneration
from pruner import QueryAwarePruner


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the visual token pruner for LLaVA using saved provenance samples. "
            "Only the pruner is updated; the vision encoder and language model stay frozen."
        )
    )
    parser.add_argument("--samples-dir", type=str, default="dataset/samples")
    parser.add_argument(
        "--manifest",
        type=str,
        default="dataset/manifest.jsonl",
        help="Optional JSONL manifest used to index sample ids and metadata before loading .pt files.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/pruner_training")
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument(
        "--target-layer",
        type=int,
        default=7,
        help="Checkpoint layer to supervise against. The trainer averages this layer's provenance across all saved decode steps.",
    )
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--soft-target-weight", type=float, default=0.25)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--single-head", action="store_true")
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="",
        help="Optional directory for cached frozen features to avoid recomputing the vision encoder each epoch.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to datasets.load_dataset for datasets with custom loaders like HuggingFaceM4/VQAv2.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    # to ensure reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_dtype(device):
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def build_model_kwargs(args, model_dtype):
    kwargs = {"low_cpu_mem_usage": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = model_dtype
    return kwargs


def load_manifest_entries(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    entries = []
    for line_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_idx} of {path}: {exc}") from exc

    return entries


def resolve_sample_path(entry, samples_dir):
    sample_id = str(entry["id"])
    local_by_id = Path(samples_dir) / f"{sample_id}.pt"
    if local_by_id.exists():
        return local_by_id

    sample_file = entry.get("sample_file")
    if sample_file:
        sample_path = Path(sample_file)
        if sample_path.exists():
            return sample_path

    raise FileNotFoundError(
        f"Could not resolve sample file for manifest id {sample_id}. "
        f"Expected {local_by_id} or an existing manifest sample_file path."
    )


def build_record_from_sample(path, sample, manifest_entry=None):
    metadata = sample.get("metadata", {})
    dataset_name = metadata.get("dataset")
    split = metadata.get("split")
    question_id = metadata.get("question_id") or sample.get("id")
    if dataset_name is None or split is None or question_id is None:
        raise ValueError(
            f"{path} is missing metadata required to recover the original image "
            "(expected dataset, split, and question_id)."
        )

    return {
        "path": path,
        "sample": sample,
        "dataset_name": dataset_name,
        "split": split,
        "question_id": str(question_id),
        "manifest_entry": manifest_entry,
    }


def load_sample_records(samples_dir, manifest_path=None):
    samples_dir = Path(samples_dir)
    if not manifest_path:
        raise ValueError("--manifest is required.")

    manifest_entries = load_manifest_entries(manifest_path)

    records = []
    for entry in manifest_entries:
        if "id" not in entry:
            raise ValueError(f"Manifest entry is missing required field 'id': {entry}")

        path = resolve_sample_path(entry, samples_dir)
        sample = torch.load(path, map_location="cpu")
        records.append(build_record_from_sample(path=path, sample=sample, manifest_entry=entry))

    print(f"Loaded sample index from manifest: {manifest_path}")
    return records


def normalize_question_id(example, fallback_idx):
    question_id = example.get("question_id")
    if question_id is None:
        question_id = example.get("id", fallback_idx)
    return str(question_id)


def build_image_lookup(records, trust_remote_code):
    grouped = {}
    for record in records:
        key = (record["dataset_name"], record["split"])
        grouped.setdefault(key, set()).add(record["question_id"])

    resolved = {}
    for (dataset_name, split), needed_ids in grouped.items():
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            trust_remote_code=trust_remote_code,
        )

        remaining = set(needed_ids)
        desc = f"Resolving images from {dataset_name}:{split}"
        for raw_idx, example in enumerate(tqdm(dataset, desc=desc, total=None)):
            question_id = normalize_question_id(example, raw_idx)
            if question_id not in remaining:
                continue

            image = example.get("image") or example.get("img") or example.get("picture")
            if image is None:
                raise ValueError(
                    f"Dataset example {question_id} from {dataset_name}:{split} has no image field."
                )

            resolved[(dataset_name, split, question_id)] = image
            remaining.remove(question_id)
            if not remaining:
                break

        if remaining:
            missing = ", ".join(sorted(list(remaining))[:10])
            raise RuntimeError(
                f"Could not recover {len(remaining)} images from {dataset_name}:{split}. "
                f"First missing ids: {missing}"
            )

    return resolved


def select_target_layer(sample, requested_layer):
    steps = sample.get("steps", [])
    if not steps:
        raise ValueError(f"Sample {sample.get('id')} has no saved decode steps.")

    first_step = steps[0]
    checkpoint_dict = first_step.get("decode_avg_provenances")
    if checkpoint_dict is None:
        raise ValueError(
            f"Sample {sample.get('id')} does not contain decode_avg_provenances."
        )

    available_layers = sorted(int(layer) for layer in checkpoint_dict.keys())
    if requested_layer not in available_layers:
        raise ValueError(
            f"Requested target layer {requested_layer} is not available for sample {sample.get('id')}. "
            f"Available layers: {available_layers}"
        )
    return requested_layer


def build_targets(sample, keep_ratio, target_layer):
    selected_layer = select_target_layer(sample, target_layer)
    per_step_rows = []
    for step in sample["steps"]:
        checkpoint_dict = step.get("decode_avg_provenances")
        if checkpoint_dict is None:
            raise ValueError(
                f"Sample {sample.get('id')} has a step without decode_avg_provenances."
            )
        per_step_rows.append(checkpoint_dict[int(selected_layer)].to(torch.float32))

    importance = torch.stack(per_step_rows, dim=0).mean(dim=0)
    importance = importance.clamp_min(0)

    num_tokens = importance.shape[0]
    keep_k = max(1, int(num_tokens * keep_ratio))
    target_indices = torch.topk(importance, k=keep_k, dim=0).indices

    binary_target = torch.zeros_like(importance)
    binary_target[target_indices] = 1.0

    soft_target = importance / importance.max().clamp_min(1e-6)

    return {
        "importance": importance,
        "soft_target": soft_target,
        "binary_target": binary_target,
        "target_indices": target_indices,
        "keep_k": keep_k,
        "target_layer": selected_layer,
    }


def freeze_non_pruner(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def prepare_inputs(model, processor, prompt, image, device):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(device)
    return inputs


@torch.no_grad()
def compute_frozen_features(model, processor, prompt, image, device):
    inputs = prepare_inputs(model, processor, prompt, image, device)
    input_ids = inputs["input_ids"].detach().cpu()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.detach().cpu()

    image_outputs = model.model.get_image_features(
        pixel_values=inputs["pixel_values"],
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        image_sizes=inputs.get("image_sizes"),
        return_dict=True,
    )

    image_features = image_outputs.pooler_output
    image_features = torch.cat(image_features, dim=0).to(model.dtype)
    if image_features.ndim == 2:
        image_features = image_features.unsqueeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_features": image_features.detach().cpu(),
    }


@dataclass
class PreparedExample:
    sample_id: str
    prompt: str
    dataset_name: str
    split: str
    question_id: str
    image_lookup_key: tuple
    sample_path: Path
    input_ids: torch.Tensor | None
    attention_mask: torch.Tensor | None
    image_features: torch.Tensor | None
    soft_target: torch.Tensor
    binary_target: torch.Tensor
    keep_k: int


class ProvenanceFeatureDataset(Dataset):
    def __init__(
        self,
        records,
        image_lookup,
        model,
        processor,
        device,
        keep_ratio,
        target_layer,
        feature_cache_dir=None,
    ):
        self.records = records
        self.image_lookup = image_lookup
        self.model = model
        self.processor = processor
        self.device = device
        self.keep_ratio = keep_ratio
        self.target_layer = target_layer
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        self.examples = [self._prepare_record(record) for record in records]

    def _cache_path(self, sample_id):
        if self.feature_cache_dir is None:
            return None
        return self.feature_cache_dir / f"{sample_id}.pt"

    def _prepare_record(self, record):
        sample = record["sample"]
        targets = build_targets(
            sample=sample,
            keep_ratio=self.keep_ratio,
            target_layer=self.target_layer,
        )
        key = (record["dataset_name"], record["split"], record["question_id"])
        cache_path = self._cache_path(str(sample["id"]))

        input_ids = None
        attention_mask = None
        image_features = None

        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            input_ids = cached["input_ids"]
            attention_mask = cached["attention_mask"]
            image_features = cached["image_features"]

        return PreparedExample(
            sample_id=str(sample["id"]),
            prompt=sample["prompt"],
            dataset_name=record["dataset_name"],
            split=record["split"],
            question_id=record["question_id"],
            image_lookup_key=key,
            sample_path=record["path"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=image_features,
            soft_target=targets["soft_target"],
            binary_target=targets["binary_target"],
            keep_k=targets["keep_k"],
        )

    def materialize_missing_cache(self):
        if self.feature_cache_dir is None:
            return

        pending = [example for example in self.examples if example.image_features is None]
        if not pending:
            return

        for example in tqdm(pending, desc="Caching frozen LLaVA features"):
            features = self._compute_features(example)
            cache_path = self._cache_path(example.sample_id)
            torch.save(features, cache_path)
            example.input_ids = features["input_ids"]
            example.attention_mask = features["attention_mask"]
            example.image_features = features["image_features"]

    def _compute_features(self, example):
        image = self.image_lookup[example.image_lookup_key]
        return compute_frozen_features(
            model=self.model,
            processor=self.processor,
            prompt=example.prompt,
            image=image,
            device=self.device,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        if example.image_features is None:
            features = self._compute_features(example)
            input_ids = features["input_ids"]
            attention_mask = features["attention_mask"]
            image_features = features["image_features"]
        else:
            input_ids = example.input_ids
            attention_mask = example.attention_mask
            image_features = example.image_features

        return {
            "sample_id": example.sample_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_features": image_features,
            "soft_target": example.soft_target,
            "binary_target": example.binary_target,
            "keep_k": example.keep_k,
        }


def split_records(records, train_ratio, seed, max_train_samples=0, max_val_samples=0):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    train_count = max(1, int(len(shuffled) * train_ratio))
    train_records = shuffled[:train_count]
    val_records = shuffled[train_count:]

    if max_train_samples > 0:
        train_records = train_records[:max_train_samples]
    if max_val_samples > 0:
        val_records = val_records[:max_val_samples]

    if not val_records:
        raise ValueError("Validation split is empty. Reduce --train-split or provide more samples.")

    return train_records, val_records


def move_batch_to_device(batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"]
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    return {
        "sample_id": batch["sample_id"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_features": batch["image_features"].to(device=device, dtype=torch.float32),
        "soft_target": batch["soft_target"].to(device=device, dtype=torch.float32),
        "binary_target": batch["binary_target"].to(device=device, dtype=torch.float32),
        "keep_k": int(batch["keep_k"]),
    }


def compute_loss(scores, binary_target, soft_target, soft_target_weight):
    scores = scores.clamp(1e-6, 1 - 1e-6)

    positives = binary_target.sum().item()
    negatives = binary_target.numel() - positives
    pos_weight = max(1.0, negatives / max(1.0, positives))
    weights = torch.where(binary_target > 0, pos_weight, 1.0).to(scores.dtype)

    bce = F.binary_cross_entropy(scores, binary_target, weight=weights)
    soft_mse = F.mse_loss(scores, soft_target)
    total = bce + (soft_target_weight * soft_mse)
    return total, bce.detach(), soft_mse.detach()


def overlap_at_k(scores, binary_target, keep_k):
    pred_idx = torch.topk(scores, k=keep_k, dim=-1).indices
    target_idx = torch.topk(binary_target, k=keep_k, dim=-1).indices
    pred_set = set(pred_idx.tolist())
    target_set = set(target_idx.tolist())
    return len(pred_set & target_set) / max(1, keep_k)


def run_epoch(dataset, model, pruner, optimizer, device, soft_target_weight, train):
    total_loss = 0.0
    total_bce = 0.0
    total_soft = 0.0
    total_overlap = 0.0
    num_batches = len(dataset)

    mode = "train" if train else "eval"
    iterator = tqdm(dataset, desc=f"{mode} epoch", total=num_batches)

    if train:
        pruner.train()
    else:
        pruner.eval()

    for batch in iterator:
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            text_embeds = model.get_input_embeddings()(batch["input_ids"]).to(torch.float32)

        if train:
            optimizer.zero_grad(set_to_none=True)
            pruner(batch["image_features"], text_embeds, keep_ratio=batch["keep_k"] / batch["binary_target"].numel())
            scores = pruner.last_importance_scores.squeeze(0)
            loss, bce, soft = compute_loss(
                scores=scores,
                binary_target=batch["binary_target"],
                soft_target=batch["soft_target"],
                soft_target_weight=soft_target_weight,
            )
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pruner(batch["image_features"], text_embeds, keep_ratio=batch["keep_k"] / batch["binary_target"].numel())
                scores = pruner.last_importance_scores.squeeze(0)
                loss, bce, soft = compute_loss(
                    scores=scores,
                    binary_target=batch["binary_target"],
                    soft_target=batch["soft_target"],
                    soft_target_weight=soft_target_weight,
                )

        overlap = overlap_at_k(scores, batch["binary_target"], batch["keep_k"])
        total_loss += float(loss.item())
        total_bce += float(bce.item())
        total_soft += float(soft.item())
        total_overlap += float(overlap)

        iterator.set_postfix(
            loss=f"{total_loss / (iterator.n or 1):.4f}",
            overlap=f"{total_overlap / (iterator.n or 1):.4f}",
        )

    denom = max(1, num_batches)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "soft_mse": total_soft / denom,
        "topk_overlap": total_overlap / denom,
    }


def save_checkpoint(output_dir, epoch, pruner, args, best_val_overlap):
    checkpoint = {
        "epoch": epoch,
        "best_val_topk_overlap": best_val_overlap,
        "model_id": args.model_id,
        "keep_ratio": args.keep_ratio,
        "target_layer": args.target_layer,
        "pruner_config": {
            "dim": pruner.dim,
            "num_heads": pruner.num_heads,
            "use_multi_head": pruner.use_multi_head,
        },
        "state_dict": pruner.state_dict(),
    }
    torch.save(checkpoint, Path(output_dir) / "best_pruner.pt")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device()
    model_dtype = resolve_model_dtype(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_sample_records(args.samples_dir, args.manifest)
    train_records, val_records = split_records(
        records=records,
        train_ratio=args.train_split,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    print(f"Loaded {len(records)} provenance samples from {args.samples_dir}")
    print(f"Train samples: {len(train_records)} | Val samples: {len(val_records)}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = PrunableLlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        **build_model_kwargs(args, model_dtype),
    ).to(device)
    freeze_non_pruner(model)

    pruner_dim = model.get_input_embeddings().embedding_dim
    pruner = QueryAwarePruner(
        dim=pruner_dim,
        num_heads=args.num_heads,
        use_multi_head=not args.single_head,
    ).to(device=device, dtype=torch.float32)
    model.pruner = pruner

    print("Resolving original images from the source dataset...")
    image_lookup = build_image_lookup(records, trust_remote_code=args.trust_remote_code)

    train_dataset = ProvenanceFeatureDataset(
        records=train_records,
        image_lookup=image_lookup,
        model=model,
        processor=processor,
        device=device,
        keep_ratio=args.keep_ratio,
        target_layer=args.target_layer,
        feature_cache_dir=Path(args.feature_cache_dir) / "train" if args.feature_cache_dir else None,
    )
    val_dataset = ProvenanceFeatureDataset(
        records=val_records,
        image_lookup=image_lookup,
        model=model,
        processor=processor,
        device=device,
        keep_ratio=args.keep_ratio,
        target_layer=args.target_layer,
        feature_cache_dir=Path(args.feature_cache_dir) / "val" if args.feature_cache_dir else None,
    )

    train_dataset.materialize_missing_cache()
    val_dataset.materialize_missing_cache()

    optimizer = torch.optim.AdamW(
        pruner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_overlap = float("-inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            dataset=train_dataset,
            model=model,
            pruner=pruner,
            optimizer=optimizer,
            device=device,
            soft_target_weight=args.soft_target_weight,
            train=True,
        )
        val_metrics = run_epoch(
            dataset=val_dataset,
            model=model,
            pruner=pruner,
            optimizer=optimizer,
            device=device,
            soft_target_weight=args.soft_target_weight,
            train=False,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)

        print(
            f"train loss={train_metrics['loss']:.4f} overlap={train_metrics['topk_overlap']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} overlap={val_metrics['topk_overlap']:.4f}"
        )

        if val_metrics["topk_overlap"] > best_val_overlap:
            best_val_overlap = val_metrics["topk_overlap"]
            save_checkpoint(output_dir, epoch, pruner, args, best_val_overlap)

    summary = {
        "model_id": args.model_id,
        "samples_dir": args.samples_dir,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "keep_ratio": args.keep_ratio,
        "target_layer": args.target_layer,
        "best_val_topk_overlap": best_val_overlap,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved best pruner checkpoint to {output_dir / 'best_pruner.pt'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
