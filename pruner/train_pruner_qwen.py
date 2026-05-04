import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
# from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# from custom_llava_new import PrunableLlavaForConditionalGeneration
# from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from pruner import QueryAwarePruner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the visual token pruner for Qwen2.5-VL using saved debiased attention samples."
    )
    # Two sample dirs
    parser.add_argument(
        "--samples-dirs",
        type=str,
        nargs="+",
        default=[
            "/content/drive/MyDrive/dynamic_pruning/DynamicVisualTokenPruning/samples_dir1",
            "/content/drive/MyDrive/dynamic_pruning/DynamicVisualTokenPruning/samples_dir2",
        ],
        help="One or more directories containing .pt sample files."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/pruner_training")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--target-layer", type=int, default=10)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--soft-target-weight", type=float, default=0.25)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--single-head", action="store_true")
    parser.add_argument("--feature-cache-dir", type=str, default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
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


IMAGE_DIR = "/content/drive/MyDrive/dynamic_pruning/DynamicPruning_Final/images"

def _load_single_dir(samples_dir):
    samples_dir = Path(samples_dir)
    records = []
    for pt_file in sorted(samples_dir.glob("*.pt")):
        sample_id = pt_file.stem.split("_")[-1]
        image_path = Path(IMAGE_DIR) / f"sample_{sample_id}.jpg"
        if not image_path.exists():
            print(f"WARNING: image not found for sample {sample_id}, skipping.")
            continue
        records.append({
            "sample_id": str(sample_id),
            "question_id": str(sample_id),
            "pt_path": pt_file,        # store path only
            "image_path": image_path,
        })
    print(f"Loaded {len(records)} samples from {samples_dir}")
    return records

def load_sample_records(samples_dirs):
    if isinstance(samples_dirs, str):
        samples_dirs = [samples_dirs]
    
    records = []
    seen_ids = set()
    
    for samples_dir in samples_dirs:
        dir_records = _load_single_dir(samples_dir)
        for r in dir_records:
            if r["sample_id"] in seen_ids:
                print(f"WARNING: duplicate sample_id {r['sample_id']}, skipping.")
                continue
            seen_ids.add(r["sample_id"])
            records.append(r)
    
    print(f"Total loaded: {len(records)} samples")
    return records


def normalize_question_id(example, fallback_idx):
    question_id = example.get("question_id")
    if question_id is None:
        question_id = example.get("id", fallback_idx)
    return str(question_id)


# def build_image_lookup(records, trust_remote_code):
#     grouped = {}
#     for record in records:
#         key = (record["dataset_name"], record["split"])
#         grouped.setdefault(key, set()).add(record["question_id"])

#     resolved = {}
#     for (dataset_name, split), needed_ids in grouped.items():
#         dataset = load_dataset(
#             dataset_name,
#             split=split,
#             streaming=True,
#             trust_remote_code=trust_remote_code,
#         )

#         remaining = set(needed_ids)
#         desc = f"Resolving images from {dataset_name}:{split}"
#         for raw_idx, example in enumerate(tqdm(dataset, desc=desc, total=None)):
#             question_id = normalize_question_id(example, raw_idx)
#             if question_id not in remaining:
#                 continue

#             image = example.get("image") or example.get("img") or example.get("picture")
#             if image is None:
#                 raise ValueError(
#                     f"Dataset example {question_id} from {dataset_name}:{split} has no image field."
#                 )

#             resolved[(dataset_name, split, question_id)] = image
#             remaining.remove(question_id)
#             if not remaining:
#                 break

#         if remaining:
#             missing = ", ".join(sorted(list(remaining))[:10])
#             raise RuntimeError(
#                 f"Could not recover {len(remaining)} images from {dataset_name}:{split}. "
#                 f"First missing ids: {missing}"
#             )

#     return resolved


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
    debiased = sample["debiased"]  # [20, 6, 345]
    importance = debiased[target_layer].mean(0).to(torch.float32)  # [345]
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
        "target_layer": target_layer,
    }


def freeze_non_pruner(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def prepare_inputs(processor, image_path, question, device):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

def vision_embeds(model, inputs):
    pixel_values = inputs["pixel_values"].type(
        model.model.visual.get_dtype() if hasattr(model.model.visual, "get_dtype")
        else inputs["pixel_values"].dtype
    )
    with torch.no_grad():
        embeds = model.model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
    return embeds.pooler_output if hasattr(embeds, "pooler_output") else embeds

@torch.no_grad()
def compute_frozen_features(model, processor, sample, image_path, device):
    inputs = prepare_inputs(processor, image_path, sample["question"], device)
    image_features = vision_embeds(model, inputs)
    if image_features.ndim == 2:
        image_features = image_features.unsqueeze(0)  # [1, N, D]

    return {
        "input_ids": inputs["input_ids"].detach().cpu(),
        "attention_mask": inputs["attention_mask"].detach().cpu(),
        "image_features": image_features.detach().cpu().to(torch.float32),
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
        model,
        processor,
        device,
        keep_ratio,
        target_layer,
        feature_cache_dir=None,
    ):
        self.records = records
        self.model = model
        self.processor = processor
        self.device = device
        self.keep_ratio = keep_ratio
        self.target_layer = target_layer
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        self.examples = [None] * len(records)
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(self._prepare_record, record): i
                for i, record in enumerate(records)
            }
            with tqdm(total=len(records), desc="Preparing records") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    self.examples[idx] = future.result()
                    pbar.update(1)

    def _cache_path(self, sample_id):
        if self.feature_cache_dir is None:
            return None
        return self.feature_cache_dir / f"{sample_id}.pt"

    # def _prepare_record(self, record):
    #     sample = record["sample"]
    #     targets = build_targets(
    #         sample=sample,
    #         keep_ratio=self.keep_ratio,
    #         target_layer=self.target_layer,
        # )
        # cache_path = self._cache_path(str(record["sample_id"]))

        # input_ids = None
        # attention_mask = None
        # image_features = None

        # if cache_path is not None and cache_path.exists():
        #     cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        #     input_ids = cached["input_ids"]
        #     attention_mask = cached["attention_mask"]
        #     image_features = cached["image_features"]

        # return PreparedExample(
        #     sample_id=str(record["sample_id"]),
        #     prompt=sample["question"],
        #     dataset_name="local",
        #     split="local",
        #     question_id=str(record["sample_id"]),
        #     image_lookup_key=str(record["sample_id"]),
        #     sample_path=record.get("path", Path("")),
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     image_features=image_features,
        #     soft_target=targets["soft_target"],
        #     binary_target=targets["binary_target"],
        #     keep_k=targets["keep_k"],
        # )
    
    def _prepare_record(self, record):
        # Load lazily — discard sample after targets built
        sample = torch.load(record["pt_path"], map_location="cpu", weights_only=False)
        targets = build_targets(
            sample=sample,
            keep_ratio=self.keep_ratio,
            target_layer=self.target_layer,
        )
        prompt = sample["question"]  # extract before discarding
        del sample  # free RAM immediately

        cache_path = self._cache_path(str(record["sample_id"]))
        input_ids = None
        attention_mask = None
        image_features = None

        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            input_ids = cached["input_ids"]
            attention_mask = cached["attention_mask"]
            image_features = cached["image_features"]

        return PreparedExample(
            sample_id=str(record["sample_id"]),
            prompt=prompt,
            dataset_name="local",
            split="local",
            question_id=str(record["sample_id"]),
            image_lookup_key=str(record["sample_id"]),
            sample_path=record["pt_path"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=image_features,
            soft_target=targets["soft_target"],
            binary_target=targets["binary_target"],
            keep_k=targets["keep_k"],
        )

    # def materialize_missing_cache(self):
    #     if self.feature_cache_dir is None:
    #         return

    #     pending = [example for example in self.examples if example.image_features is None]
    #     if not pending:
    #         return

    #     for example in tqdm(pending, desc="Caching frozen Qwen features"):
    #         features = self._compute_features(example)
    #         cache_path = self._cache_path(example.sample_id)
    #         torch.save(features, cache_path)
    #         example.input_ids = features["input_ids"]
    #         example.attention_mask = features["attention_mask"]
    #         example.image_features = features["image_features"]
    
    def materialize_missing_cache(self):
        if self.feature_cache_dir is None:
            return

        pending = [example for example in self.examples if example.image_features is None]
        if not pending:
            return

        record_map = {str(r["sample_id"]): r for r in self.records}

        for example in tqdm(pending, desc="Caching frozen Qwen features"):
            record = record_map[example.sample_id]
            sample = torch.load(record["pt_path"], map_location="cpu", weights_only=False)
            question = sample["question"]
            del sample

            features = compute_frozen_features(
                model=self.model,
                processor=self.processor,
                sample={"question": question},
                image_path=record["image_path"],
                device=self.device,
            )

            cache_path = self._cache_path(example.sample_id)
            torch.save(features, cache_path)
            example.input_ids = features["input_ids"]
            example.attention_mask = features["attention_mask"]
            example.image_features = features["image_features"]


    # def _compute_features(self, example):
    #     record = next(r for r in self.records if str(r["sample_id"]) == example.sample_id)
    #     return compute_frozen_features(
    #         model=self.model,
    #         processor=self.processor,
    #         sample=record["sample"],
    #         image_path=record["image_path"],
    #         device=self.device,
    #     )
    
    def _compute_features(self, example):
        record = next(r for r in self.records if str(r["sample_id"]) == example.sample_id)
        sample = torch.load(record["pt_path"], map_location="cpu", weights_only=False)
        return compute_frozen_features(
            model=self.model,
            processor=self.processor,
            sample=sample,
            image_path=record["image_path"],
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
            # text_embeds = model.get_input_embeddings()(batch["input_ids"]).to(torch.float32)
            text_embeds = model.model.language_model.embed_tokens(batch["input_ids"]).to(torch.float32)

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


def save_checkpoint(output_dir, epoch, pruner, optimizer, args, best_val_overlap, history, train_records, val_records):
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
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "train_indices": [r["question_id"] for r in train_records],
        "val_indices": [r["question_id"] for r in val_records],
    }
    torch.save(checkpoint, Path(output_dir) / "best_pruner.pt")

def load_checkpoint(output_dir, pruner, optimizer, device):
    for filename in ["last_pruner.pt", "best_pruner.pt"]:
        path = Path(output_dir) / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=device)
            pruner.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_overlap = checkpoint["best_val_topk_overlap"]
            history = checkpoint.get("history", [])
            train_indices = checkpoint.get("train_indices")
            val_indices = checkpoint.get("val_indices")
            print(f"Resuming from {filename}, epoch {start_epoch}, best val overlap: {best_val_overlap:.4f}")
            return start_epoch, best_val_overlap, history, train_indices, val_indices

    return 1, float("-inf"), [], None, None

def read_checkpoint(output_dir, device):
    """Just reads the checkpoint file, no side effects."""
    for filename in ["last_pruner.pt", "best_pruner.pt"]:
        path = Path(output_dir) / filename
        if path.exists():
            print(f"Found checkpoint: {filename}")
            return torch.load(path, map_location=device)
    return None

def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device()
    model_dtype = resolve_model_dtype(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume or fresh start
    start_epoch = 1
    best_val_overlap = float("-inf")
    history = []
    train_indices = None
    val_indices = None

    if args.resume:
        start_epoch, best_val_overlap, history, train_indices, val_indices = load_checkpoint(
            args.output_dir, pruner, optimizer, device
        )

    records = load_sample_records(args.samples_dir, args.manifest)
    print(f"Loaded {len(records)} provenance samples from {args.samples_dir}")

    # Reconstruct exact same split if resuming, otherwise create fresh split
    if train_indices is not None:
        index_map = {r["question_id"]: r for r in records}
        train_records = [index_map[i] for i in train_indices if i in index_map]
        val_records = [index_map[i] for i in val_indices if i in index_map]
        print(f"Restored split: {len(train_records)} train | {len(val_records)} val")
    else:
        train_records, val_records = split_records(
            records=records,
            train_ratio=args.train_split,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
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

    # history = []
    # best_val_overlap = float("-inf")

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
            save_checkpoint(output_dir, epoch, pruner, optimizer, args, best_val_overlap, history, train_records, val_records)

        # Save last checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "best_val_topk_overlap": best_val_overlap,
            "state_dict": pruner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "train_indices": [r["question_id"] for r in train_records],
            "val_indices": [r["question_id"] for r in val_records],
        }, output_dir / "last_pruner.pt")

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
